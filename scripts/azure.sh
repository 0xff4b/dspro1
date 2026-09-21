#!/usr/bin/env bash
# Run from WSL/Linux: bash scripts/azure.sh help
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-f896f8e6-927d-4630-8d87-fdcead25030e}"
RG="${AZURE_RESOURCE_GROUP:-rg-dspro1-prototype}"
LOCATION="${AZURE_LOCATION:-switzerlandnorth}"
APP="${AZURE_APP_NAME:-dspro1-streamlit}"
ENVIRONMENT="${AZURE_ENVIRONMENT_NAME:-dspro1-prototype}"
SUFFIX="$(printf '%s' "$SUBSCRIPTION_ID/$RG" | sha256sum | cut -c1-12)"
ACR="${AZURE_REGISTRY_NAME:-dspro1$SUFFIX}"
OWNER_TAG="dspro1-prototype-v1"
azc() { bash "$ROOT/scripts/az.sh" "$@" --subscription "$SUBSCRIPTION_ID"; }
fail() { echo "$*" >&2; exit 1; }
owned_group() {
    [[ "$(azc group show -n "$RG" --query tags.managedBy -o tsv)" == "$OWNER_TAG" ]] ||
        fail "Abbruch: $RG gehört nicht zu diesem Deployment."
}
status() {
    azc containerapp show -g "$RG" -n "$APP" \
        --query '{url:properties.configuration.ingress.fqdn,state:properties.provisioningState,min:properties.template.scale.minReplicas,max:properties.template.scale.maxReplicas}' -o table
    azc containerapp revision list -g "$RG" -n "$APP" \
        --query '[].{revision:name,active:properties.active,health:properties.healthState}' -o table
}
case "${1:-help}" in
    help)
        echo "bash scripts/azure.sh login|deploy|status|start|stop|delete --yes"
        echo "deploy: lokales Docker-Image bauen, testen, hochladen und starten."
        echo "Standard: private ACR Basic (kostenpflichtig), Consumption mit 0–1 Instanzen."
        echo "DEPLOY_IMAGE=ghcr.io/...:tag verwendet ein bereits hochgeladenes Image ohne ACR."
        exit 0 ;;
    login)
        bash "$ROOT/scripts/az.sh" login --use-device-code --output none
        azc account show --query '{subscription:id,name:name}' -o table
        exit 0 ;;
    deploy|status|start|stop|delete) ;;
    *) fail "Unbekannter Befehl. Verwende help." ;;
esac
azc account show --query id -o tsv >/dev/null ||
    fail "Azure-Anmeldung erforderlich: bash scripts/azure.sh login"
if [[ "$1" != deploy ]]; then
    owned_group
    case "$1" in
        status) status ;;
        stop)
            revisions="$(azc containerapp revision list -g "$RG" -n "$APP" --query '[?properties.active].name' -o tsv)"
            while IFS= read -r revision; do
                [[ -z "$revision" ]] || azc containerapp revision deactivate -g "$RG" -n "$APP" --revision "$revision" -o none
            done <<< "$revisions"
            echo "App gestoppt. Eine eventuell vorhandene Registry bleibt kostenpflichtig." ;;
        start)
            revision="$(azc containerapp show -g "$RG" -n "$APP" --query properties.latestReadyRevisionName -o tsv)"
            [[ -n "$revision" ]] || fail "Keine startbereite Revision vorhanden. Erneut deploy ausführen."
            azc containerapp revision activate -g "$RG" -n "$APP" --revision "$revision" -o none
            status ;;
        delete)
            [[ "${2:-}" == --yes ]] || fail "Löscht ALLE Ressourcen in $RG. Bestätigen mit: bash scripts/azure.sh delete --yes"
            azc resource list -g "$RG" --query '[].{name:name,type:type}' -o table
            azc group delete -n "$RG" --yes
            echo "Ressourcengruppe $RG gelöscht. Externe GHCR-Images und CI-Identität bleiben erhalten." ;;
    esac
    exit 0
fi

# Build and test before provisioning any chargeable resources.
IMAGE="${DEPLOY_IMAGE:-}"
if [[ -z "$IMAGE" ]]; then
    TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M%S)}"
    docker build --platform linux/amd64 -t "dspro1-streamlit:$TAG" .
    docker run --rm --memory 512m --cpus 0.25 \
        -v "$ROOT/scripts/smoke_app.py:/tmp/smoke_app.py:ro" \
        "dspro1-streamlit:$TAG" python /tmp/smoke_app.py
fi
if [[ "${AZURE_SKIP_PROVIDER_REGISTRATION:-0}" != 1 ]]; then
for provider in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.ManagedIdentity; do
    azc provider register --namespace "$provider" --wait -o none
done
fi
if [[ "$(azc group exists -n "$RG" -o tsv)" == true ]]; then
    owned_group
else
    azc group create -n "$RG" -l "$LOCATION" --tags managedBy="$OWNER_TAG" purpose=prototype -o none
fi
# Errors querying resources must fail, not be mistaken for absence.
ENV_ID="$(azc resource list -g "$RG" --resource-type Microsoft.App/managedEnvironments --query "[?name=='$ENVIRONMENT'].id | [0]" -o tsv)"
if [[ -z "$ENV_ID" ]]; then
    azc containerapp env create -g "$RG" -n "$ENVIRONMENT" -l "$LOCATION" \
        --enable-workload-profiles false --logs-destination none -o none
fi
registry_args=()
if [[ -z "$IMAGE" ]]; then
    ACR_ID="$(azc resource list -g "$RG" --resource-type Microsoft.ContainerRegistry/registries --query "[?name=='$ACR'].id | [0]" -o tsv)"
    if [[ -z "$ACR_ID" ]]; then
        azc acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled false -o none
    fi
    SERVER="$(azc acr show -g "$RG" -n "$ACR" --query loginServer -o tsv)"
    ACR_ID="$(azc acr show -g "$RG" -n "$ACR" --query id -o tsv)"
    # Credentials are passed through stdin and stored only in a temporary Docker config.
    DOCKER_AUTH="$(mktemp -d)"
    trap 'rm -rf -- "$DOCKER_AUTH"' EXIT
    azc acr login -n "$ACR" --expose-token --query accessToken -o tsv |
        docker --config "$DOCKER_AUTH" login "$SERVER" --username 00000000-0000-0000-0000-000000000000 --password-stdin
    IMAGE="$SERVER/dspro1:$TAG"
    docker tag "dspro1-streamlit:$TAG" "$IMAGE"
    docker --config "$DOCKER_AUTH" push "$IMAGE"
    IDENTITY_ID="$(azc identity create -g "$RG" -n dspro1-pull --query id -o tsv)"
    PRINCIPAL="$(azc identity show -g "$RG" -n dspro1-pull --query principalId -o tsv)"
    azc role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
        --role AcrPull --scope "$ACR_ID" -o none
    registry_args=(--user-assigned "$IDENTITY_ID" --registry-server "$SERVER" --registry-identity "$IDENTITY_ID")
elif [[ -n "${REGISTRY_USERNAME:-}" || -n "${REGISTRY_PASSWORD:-}" ]]; then
    [[ -n "${REGISTRY_USERNAME:-}" && -n "${REGISTRY_PASSWORD:-}" ]] || fail "Beide Registry-Zugangsdaten setzen."
    registry_args=(--registry-server "${IMAGE%%/*}" --registry-username "$REGISTRY_USERNAME" --registry-password "$REGISTRY_PASSWORD")
fi
# create is idempotent and reconciles ingress, identity and resource limits on updates.
azc containerapp create -g "$RG" -n "$APP" --environment "$ENVIRONMENT" \
    --image "$IMAGE" --ingress external --target-port 8501 --transport auto \
    --cpu 0.25 --memory 0.5Gi --min-replicas 0 --max-replicas 1 \
    --scale-rule-name http --scale-rule-type http --scale-rule-http-concurrency 10 \
    --revisions-mode single "${registry_args[@]}" -o none
URL="https://$(azc containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)"
curl --fail --silent --show-error --retry 20 --retry-all-errors --retry-delay 10 \
    --connect-timeout 10 --max-time 30 "$URL/_stcore/health"
printf '\nBereit: %s\n' "$URL"

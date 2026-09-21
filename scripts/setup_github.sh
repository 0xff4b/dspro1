#!/usr/bin/env bash
# Once per deployment resource group; grants CI access only to that group.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPO="${1:?Usage: bash scripts/setup_github.sh OWNER/REPOSITORY}"
[[ "$REPO" =~ ^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$ ]] || exit 1
SUB="${AZURE_SUBSCRIPTION_ID:-f896f8e6-927d-4630-8d87-fdcead25030e}"
RG="${AZURE_RESOURCE_GROUP:-rg-dspro1-prototype}"
LOCATION="${AZURE_LOCATION:-switzerlandnorth}"
IDENTITY_RG="rg-dspro1-automation"
azc() { bash scripts/az.sh "$@" --subscription "$SUB"; }
command -v gh >/dev/null || { echo "GitHub CLI installieren und gh auth login ausführen."; exit 1; }
gh repo view "$REPO" --json nameWithOwner >/dev/null
azc account show -o none
for provider in Microsoft.App Microsoft.OperationalInsights Microsoft.ManagedIdentity; do
    azc provider register --namespace "$provider" --wait -o none
done
for group in "$RG" "$IDENTITY_RG"; do
    if [[ "$(azc group exists -n "$group" -o tsv)" == true ]]; then
        [[ "$(azc group show -n "$group" --query tags.managedBy -o tsv)" == dspro1-prototype-v1 ]] || { echo "Fremde Ressourcengruppe: $group"; exit 1; }
    else
        azc group create -n "$group" -l "$LOCATION" --tags managedBy=dspro1-prototype-v1 -o none
    fi
done
NAME="dspro1-github-$(printf '%s' "$REPO" | sha256sum | cut -c1-8)"
azc identity create -g "$IDENTITY_RG" -n "$NAME" -o none
CLIENT="$(azc identity show -g "$IDENTITY_RG" -n "$NAME" --query clientId -o tsv)"
PRINCIPAL="$(azc identity show -g "$IDENTITY_RG" -n "$NAME" --query principalId -o tsv)"
TENANT="$(azc account show --query tenantId -o tsv)"
azc identity federated-credential create -g "$IDENTITY_RG" --identity-name "$NAME" -n github-main \
    --issuer https://token.actions.githubusercontent.com \
    --subject "repo:$REPO:ref:refs/heads/main" --audiences api://AzureADTokenExchange -o none
azc role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
    --role Contributor --scope "/subscriptions/$SUB/resourceGroups/$RG" -o none
gh variable set AZURE_CLIENT_ID -R "$REPO" --body "$CLIENT"
gh variable set AZURE_TENANT_ID -R "$REPO" --body "$TENANT"
gh variable set AZURE_SUBSCRIPTION_ID -R "$REPO" --body "$SUB"
gh variable set AZURE_RESOURCE_GROUP -R "$REPO" --body "$RG"
gh variable set AZURE_LOCATION -R "$REPO" --body "$LOCATION"
gh variable set AZURE_AUTO_DEPLOY -R "$REPO" --body false
echo "OIDC eingerichtet. GHCR_USERNAME und Secret GHCR_READ_TOKEN (classic PAT, read:packages) setzen."
echo "Danach manuell den Workflow starten; für Deploy bei Push AZURE_AUTO_DEPLOY=true setzen."

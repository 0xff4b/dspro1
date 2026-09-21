"""Safety checks for the Azure lifecycle commands using a fake Azure CLI."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAKE_AZ = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$AZURE_TEST_LOG"
case "$1 $2" in
  "account show") echo subscription ;;
  "group show") echo "$AZURE_TEST_OWNER" ;;
  "containerapp revision")
    case "$3" in
      list) printf 'app--one\napp--two\n' ;;
      activate|deactivate) ;;
    esac ;;
  "containerapp show")
    if [[ "$*" == *latestReadyRevisionName* ]]; then echo app--ready; fi ;;
  "resource list") echo resources ;;
  "group delete") ;;
  *) echo "Unexpected command" >&2; exit 99 ;;
esac
"""


class LifecycleTests(unittest.TestCase):
    def invoke(self, *args, owner="dspro1-prototype-v1"):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "az").write_text(FAKE_AZ)
            (path / "az").chmod(0o755)
            env = os.environ | {
                "PATH": folder + os.pathsep + os.environ["PATH"],
                "AZURE_TEST_LOG": str(path / "calls"),
                "AZURE_TEST_OWNER": owner,
            }
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/azure.sh"), *args],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
            calls = (path / "calls").read_text()
            return result, calls

    def test_foreign_group_cannot_be_deleted(self):
        result, calls = self.invoke("delete", "--yes", owner="someone-else")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("group delete", calls)

    def test_delete_requires_explicit_confirmation(self):
        result, calls = self.invoke("delete")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("group delete", calls)

    def test_delete_targets_only_configured_subscription_and_group(self):
        result, calls = self.invoke("delete", "--yes")
        self.assertEqual(result.returncode, 0, result.stderr)
        delete = next(line for line in calls.splitlines() if line.startswith("group delete"))
        self.assertIn("-n rg-dspro1-prototype --yes", delete)
        self.assertIn("--subscription f896f8e6-927d-4630-8d87-fdcead25030e", delete)

    def test_stop_deactivates_active_revisions(self):
        result, calls = self.invoke("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[?properties.active].name", calls)
        self.assertEqual(calls.count("revision deactivate"), 2)
        self.assertNotIn("group delete", calls)

    def test_start_activates_latest_ready_revision(self):
        result, calls = self.invoke("start")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("properties.latestReadyRevisionName", calls)
        self.assertIn("--revision app--ready", calls)


if __name__ == "__main__":
    unittest.main()

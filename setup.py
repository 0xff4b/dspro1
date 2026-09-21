#!/usr/bin/env python3
"""Safe, repeatable local setup. Requires Git and 64-bit Python 3.12."""
from __future__ import annotations

import argparse
import functools
import ast
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import venv

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent
PIP_URL = "https://files.pythonhosted.org/packages/f3/6e/1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/pip-26.2.1-py3-none-any.whl"
PIP_SHA256 = "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"
PIP_RUNNER = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('pip',run_name='__main__')"


class SetupError(RuntimeError):
    pass


def run(args, *, root=ROOT, capture=False, check=True, env=None):
    result = subprocess.run(
        [str(x) for x in args], cwd=root, env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and result.returncode:
        details = result.stderr.decode(errors="replace") if capture else ""
        raise SetupError(f"Command failed ({result.returncode}): {args[0]}\n{details}")
    return result


def git(*args, root=ROOT, check=True):
    return run(["git", *args], root=root, capture=True, check=check)


def safe_path(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise SetupError(f"Refusing to modify a symlink or path outside the project: {path}")
    return path


def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def inspect_file(path):
    if not path.is_file():
        return "missing file"
    if path.suffix not in {".ipynb", ".py", ".joblib", ".csv"}:
        return None
    with path.open("rb") as stream:
        prefix = stream.read(256)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        return "Git LFS pointer instead of file contents; run git lfs pull"
    if not prefix:
        return "empty source/data/model file"
    try:
        if path.suffix == ".ipynb":
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("nbformat") != 4 or not isinstance(data.get("cells"), list):
                return "invalid notebook structure"
        elif path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        elif path.name in {"model.csv", "model_wide.csv"}:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = csv.DictReader(stream)
                if not ({"price", "area", "rooms"}.issubset(rows.fieldnames or []) or
                        {"price_cold", "area_sqm", "rooms"}.issubset(rows.fieldnames or [])):
                    return "required CSV columns missing"
                if next(rows, None) is None:
                    return "CSV has no data rows"
    except (ValueError, AttributeError, SyntaxError, UnicodeError, csv.Error) as exc:
        return f"invalid or unfinished local file: {exc}"
    return None


def inspect_repository(root=ROOT, repair=False, allow_local_data=False):
    if shutil.which("git") is None:
        raise SetupError("Install Git first: https://git-scm.com/downloads")
    if git("ls-files", "-u", root=root).stdout:
        raise SetupError("Unresolved Git merge conflicts. Finish or abort the merge before setup.")
    check = git("fsck", "--full", "--no-reflogs", root=root, check=False)
    if check.returncode:
        raise SetupError(
            "Git's object database failed validation. Keep this working copy and clone "
            "the repository into a NEW directory; --repair cannot repair broken Git objects.\n"
            + check.stderr.decode(errors="replace")
        )
    names = git("ls-files", "-z", root=root).stdout.decode().split("\0")
    issues = []
    for name in filter(None, names):
        problem = inspect_file(root / name)
        if not problem and not allow_local_data and (name.endswith(".joblib") or Path(name).name in {"model.csv", "model_wide.csv"}):
            expected = git("rev-parse", "HEAD:" + name, root=root, check=False)
            actual = git("hash-object", "--path=" + name, name, root=root)
            if expected.returncode == 0 and expected.stdout.strip() != actual.stdout.strip():
                problem = "changed/damaged data or model; use --allow-local-data for intentional training changes"
        if problem:
            issues.append((name, problem))
    if repair and issues:
        backup_root = root / ".setup-backups" / ("files-" + stamp())
        for name, problem in issues:
            path = safe_path(root, name)
            if git("diff", "--cached", "--quiet", "--", name, root=root, check=False).returncode:
                raise SetupError(f"Refusing to overwrite a staged change: {name}")
            original = git("show", "HEAD:" + name, root=root, check=False)
            if original.returncode:
                raise SetupError(f"No committed recovery copy for {name}. Local work was kept.")
            if path.exists():
                backup = backup_root / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
            git("restore", "--source=HEAD", "--worktree", "--", name, root=root)
            print(f"Restored {name}; previous local contents kept under {backup_root}")
        return inspect_repository(root=root, repair=False, allow_local_data=allow_local_data)
    if issues:
        message = "\n".join(f"  {name}: {problem}" for name, problem in issues)
        raise SetupError(
            "Incomplete files or local edits need attention:\n" + message +
            "\nTo recover these files from HEAD with backups: python setup.py --repair\n"
            "This never uses git reset --hard or git clean."
        )
    changed = git("diff", "--name-only", "HEAD", "--", root=root).stdout.decode().strip()
    if changed:
        print("Local changes detected and preserved (changes do not necessarily mean corruption).")
    print("Repository: Git objects, tracked files, notebooks and input CSVs checked.")


def python_in(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def environment_valid(environment):
    executable = python_in(environment)
    if not executable.is_file():
        return False
    probe = "import json,sys; print(json.dumps([list(sys.version_info[:2]),sys.prefix,sys.platform]))"
    try:
        result = subprocess.run([str(executable), "-I", "-c", probe], capture_output=True, timeout=20)
        version, prefix, platform = json.loads(result.stdout)
        return result.returncode == 0 and version == [3, 12] and Path(prefix).resolve() == environment.resolve() and platform == sys.platform
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def prepare_environment(root=ROOT, recreate=False):
    environment = safe_path(root, ".venv")
    if environment.exists() and (recreate or not environment_valid(environment)):
        backup = root / (".venv.backup-" + stamp())
        environment.rename(backup)
        print(f"Previous environment preserved at {backup}")
    if not environment.exists():
        # Works on minimal WSL installs that don't include ensurepip.
        venv.EnvBuilder(with_pip=False).create(environment)
        print(f"Created {environment}")
    else:
        print(f"Reusing {environment}")
    return python_in(environment)


def pip_command(executable, root=ROOT):
    if run([executable, "-m", "pip", "--version"], root=root, capture=True, check=False).returncode == 0:
        return [executable, "-m", "pip"]
    cache = root / ".setup-cache"
    cache.mkdir(exist_ok=True)
    wheel = cache / "pip-26.2.1-py3-none-any.whl"
    if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != PIP_SHA256:
        print("Bootstrapping pip inside the project environment...")
        with urllib.request.urlopen(PIP_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != PIP_SHA256:
            raise SetupError("Pip bootstrap checksum mismatch; nothing was installed.")
        temp = wheel.with_suffix(".download")
        temp.write_bytes(content)
        temp.replace(wheel)
    command = [executable, "-I", "-c", PIP_RUNNER, wheel]
    run([*command, "install", "--disable-pip-version-check", "--no-input", "--no-index", wheel], root=root)
    return [executable, "-m", "pip"]


def install(executable, profile, root=ROOT):
    command = pip_command(executable, root)
    lock = root / ("requirements-app.lock" if profile == "app" else "requirements-full.lock")
    print(f"Installing the locked {profile} environment (first full install can take several minutes)...")
    run([*command, "install", "--disable-pip-version-check", "--no-input",
         "--require-hashes", "--only-binary=:all:", "-r", lock], root=root)
    result = run([*command, "check"], root=root, check=False)
    if result.returncode:
        raise SetupError("Existing extra packages conflict. Run python setup.py --recreate-env "
                         "to keep the old environment as a backup and create a clean one.")


def validate_runtime(executable, profile, root=ROOT):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg",
               STREAMLIT_BROWSER_GATHER_USAGE_STATS="false", PYTHONUTF8="1")
    try:
        run([executable, root / "scripts/smoke_app.py"], root=root, env=env)
        if profile == "full":
            run([executable, "-c",
                 "import jupyterlab,ipykernel,seaborn,xgboost,sqlalchemy,psycopg2; "
                 "print('PASS notebook and database client imports')"], root=root, env=env)
            run([executable, "-m", "ipykernel", "install", "--sys-prefix",
                 "--name", "dspro", "--display-name", "DSPRO (.venv, Python 3.12)"], root=root)
    except SetupError as exc:
        raise SetupError(
            "Runtime/model check failed. Check the error above. On macOS LightGBM needs "
            "'brew install libomp'; on Linux it needs libgomp1. Re-run with --recreate-env "
            "if the existing environment is broken. Local models may also be incomplete; "
            "keep a backup before restoring them from Git.\n" + str(exc)
        ) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full", "app"], default="full")
    parser.add_argument("--start", choices=["app", "notebook"])
    parser.add_argument("--doctor", action="store_true", help="Read-only repository/environment checks")
    parser.add_argument("--repair", action="store_true", help="Restore invalid/missing tracked files from HEAD, keeping backups")
    parser.add_argument("--recreate-env", action="store_true", help="Back up .venv and create a fresh environment")
    parser.add_argument("--allow-local-data", action="store_true", help="Allow intentionally changed model/CSV files; prediction checks still run")
    parser.add_argument("--update", action="store_true", help="Fast-forward pull only when the working tree is clean")
    args = parser.parse_args(argv)
    if args.doctor and (args.repair or args.recreate_env or args.update or args.start):
        parser.error("--doctor cannot be combined with options that modify or start anything")
    if args.profile == "app" and args.start == "notebook":
        parser.error("--start notebook requires --profile full")
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise SetupError("Use Python 3.12 (64-bit). Windows: py -3.12 setup.py; Linux/macOS: python3.12 setup.py")
    os.chdir(ROOT)
    if args.update:
        if git("status", "--porcelain").stdout:
            raise SetupError("Local work exists. Commit/stash it yourself before --update; nothing was changed.")
        git("pull", "--ff-only")
        # A pull may update this script; continue with the new version.
        remaining = [arg for arg in (sys.argv[1:] if argv is None else argv) if arg != "--update"]
        return run([sys.executable, ROOT / "setup.py", *remaining], check=False).returncode
    inspect_repository(repair=args.repair, allow_local_data=args.allow_local_data)
    if args.doctor:
        if not environment_valid(ROOT / ".venv"):
            raise SetupError("No compatible project .venv. Run python setup.py.")
        executable = python_in(ROOT / ".venv")
        run([executable, "-m", "pip", "check"])
        print("Doctor checks passed. To verify predictions as well, run setup without --doctor.")
        return 0
    executable = prepare_environment(recreate=args.recreate_env)
    install(executable, args.profile)
    validate_runtime(executable, args.profile)
    print("\nSetup ready. No activation needed. Environment: " + str(executable))
    if args.start == "app":
        return run([executable, "-m", "streamlit", "run", ROOT / "src/app.py",
                    "--server.address=127.0.0.1"], check=False).returncode
    if args.start == "notebook":
        return run([executable, "-m", "jupyterlab", "--ServerApp.ip=127.0.0.1",
                    "--notebook-dir", ROOT,
                    ROOT / "src/notebooks/model_v3_clean.ipynb"], check=False).returncode
    print("Next: python setup.py --start app  OR  python setup.py --start notebook")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SetupError, OSError, urllib.error.URLError) as error:
        print(f"\nSETUP STOPPED: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)

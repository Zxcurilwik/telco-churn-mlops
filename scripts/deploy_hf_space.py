"""Deploy the service to a Hugging Face Docker Space.

    HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py --space <user>/<space-name> --wait

Uploads only what the Docker build needs (the Space trains the model inside
`docker build`, exactly like CI does), creating the Space on first run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FILES = ["Dockerfile", ".dockerignore", "requirements.txt", "requirements-train.txt"]
DIRS = ["src", "app", "configs"]
SPACE_README = ROOT / "deploy" / "huggingface" / "README.md"
FAILED = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "local"


def stage_files(target: Path) -> None:
    for name in FILES:
        shutil.copy2(ROOT / name, target / name)
    for name in DIRS:
        shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(SPACE_README, target / "README.md")


def space_url(space_id: str) -> str:
    owner, name = space_id.split("/")
    sub = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{sub}.hf.space"


def wait_for_space(api: HfApi, space_id: str, timeout: int, grace: int = 90) -> None:
    """Wait for the rebuild triggered by our commit, then for the app to be RUNNING.

    Right after the upload the old container may still report RUNNING, so we only
    accept RUNNING after a build/start phase was seen (or after a grace period).
    """
    started = time.monotonic()
    seen_rebuild = False
    while (elapsed := time.monotonic() - started) < timeout:
        stage = api.get_space_runtime(space_id).stage
        stage = getattr(stage, "value", stage)
        print(f"  [{elapsed:4.0f}s] space stage: {stage}", flush=True)
        if stage in FAILED:
            raise SystemExit(f"Space failed with stage {stage}; check the build logs on the Hub")
        if "BUILDING" in stage or "STARTING" in stage:
            seen_rebuild = True
        elif stage == "RUNNING" and (seen_rebuild or elapsed > grace):
            return
        time.sleep(15)
    raise SystemExit(f"Space did not reach RUNNING within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default=os.getenv("HF_SPACE_ID"), help="<user>/<space-name>")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    parser.add_argument("--wait", action="store_true", help="block until the Space is running")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not args.space or not token:
        sys.exit("Set HF_TOKEN and pass --space <user>/<space-name> (or set HF_SPACE_ID)")

    api = HfApi(token=token)
    api.create_repo(
        args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )
    sha = git_sha()
    with tempfile.TemporaryDirectory() as tmp:
        stage_files(Path(tmp))
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Deploy {sha}",
            delete_patterns=["*"],  # remove files that no longer exist in the project
        )
    print(f"Uploaded {sha} to https://huggingface.co/spaces/{args.space}")

    if args.wait:
        wait_for_space(api, args.space, args.timeout)
    print(f"APP_URL={space_url(args.space)}")
    if github_output := os.getenv("GITHUB_OUTPUT"):
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"url={space_url(args.space)}\n")


if __name__ == "__main__":
    main()

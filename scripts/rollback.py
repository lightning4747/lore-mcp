"""Automated Cloud Run rollback script (Task 18.3, 18.5).

Routes 100% of service traffic back to the previous healthy revision
when a deployment or post-deployment smoke test fails.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lore_mcp.scripts.rollback")


def get_current_and_previous_revisions(
    service: str,
    region: str,
    project: str | None = None,
) -> tuple[str | None, str | None]:
    """Query Cloud Run for the latest revisions of a service.

    Returns (latest_revision, previous_revision).
    """
    resolved_project = (
        project or os.environ.get("PROJECT_ID") or os.environ.get("GCP_PROJECT_ID")
    )
    cmd = [
        "gcloud",
        "run",
        "revisions",
        "list",
        f"--service={service}",
        f"--region={region}",
        "--format=value(name)",
        "--limit=3",
    ]
    if resolved_project:
        cmd.append(f"--project={resolved_project}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        revisions = [
            line.strip() for line in result.stdout.strip().splitlines() if line.strip()
        ]
        if not revisions:
            return None, None
        latest = revisions[0]
        prev = revisions[1] if len(revisions) > 1 else None
        return latest, prev
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to query Cloud Run revisions: %s", exc.stderr)
        return None, None
    except FileNotFoundError:
        logger.error("gcloud CLI not found in PATH")
        return None, None


def rollback_traffic(
    service: str,
    region: str,
    target_revision: str,
    project: str | None = None,
) -> bool:
    """Route 100% of service traffic to target revision."""
    resolved_project = (
        project or os.environ.get("PROJECT_ID") or os.environ.get("GCP_PROJECT_ID")
    )
    cmd = [
        "gcloud",
        "run",
        "services",
        "update-traffic",
        service,
        f"--region={region}",
        f"--to-revisions={target_revision}=100",
        "--quiet",
    ]
    if resolved_project:
        cmd.append(f"--project={resolved_project}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(
            "Successfully routed 100%% traffic for %s to revision %s",
            service,
            target_revision,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Failed to update traffic to revision %s: %s",
            target_revision,
            exc.stderr,
        )
        return False
    except FileNotFoundError:
        logger.error("gcloud CLI not found in PATH")
        return False


def run_rollback(
    service: str,
    region: str,
    target_revision: str | None = None,
    project: str | None = None,
) -> int:
    """Execute rollback to specified or discovered previous revision."""
    if not target_revision:
        logger.info(
            "No target revision specified, discovering previous revision "
            "for %s in %s...",
            service,
            region,
        )
        _latest, previous = get_current_and_previous_revisions(
            service=service,
            region=region,
            project=project,
        )
        if not previous:
            logger.error(
                "Cannot perform rollback: no previous revision found for service '%s'",
                service,
            )
            return 1
        target_revision = previous

    logger.warning(
        "Initiating rollback of service '%s' to revision '%s'...",
        service,
        target_revision,
    )
    success = rollback_traffic(
        service=service,
        region=region,
        target_revision=target_revision,
        project=project,
    )
    if success:
        logger.info("Rollback completed successfully.")
        return 0
    else:
        logger.error("Rollback execution failed.")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollback Cloud Run service to previous revision"
    )
    parser.add_argument(
        "--service",
        default="lore-mcp",
        help="Cloud Run service name (default: lore-mcp)",
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        help="Google Cloud region (default: us-central1)",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PROJECT_ID") or os.environ.get("GCP_PROJECT_ID"),
        help="Google Cloud project ID (optional, defaults to PROJECT_ID env)",
    )
    parser.add_argument(
        "--to-revision",
        default=None,
        help=(
            "Specific revision to roll back to "
            "(optional, discovers previous if omitted)"
        ),
    )

    args = parser.parse_args()
    code = run_rollback(
        service=args.service,
        region=args.region,
        target_revision=args.to_revision,
        project=args.project,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

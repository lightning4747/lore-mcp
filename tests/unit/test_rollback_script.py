"""Unit tests for scripts/rollback.py (Task 18.3, 18.5)."""

import subprocess
from unittest.mock import patch

import pytest
from scripts.rollback import (
    get_current_and_previous_revisions,
    rollback_traffic,
    run_rollback,
)


@pytest.mark.unit
def test_get_current_and_previous_revisions_success() -> None:
    mock_stdout = "lore-mcp-00003-abc\nlore-mcp-00002-xyz\nlore-mcp-00001-old\n"
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["gcloud"], returncode=0, stdout=mock_stdout
        ),
    ) as mock_run:
        latest, prev = get_current_and_previous_revisions(
            service="lore-mcp", region="us-central1", project="my-project"
        )
        assert latest == "lore-mcp-00003-abc"
        assert prev == "lore-mcp-00002-xyz"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "--service=lore-mcp" in cmd
        assert "--region=us-central1" in cmd
        assert "--project=my-project" in cmd


@pytest.mark.unit
def test_get_current_and_previous_revisions_single() -> None:
    mock_stdout = "lore-mcp-00001-first\n"
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["gcloud"], returncode=0, stdout=mock_stdout
        ),
    ):
        latest, prev = get_current_and_previous_revisions(
            service="lore-mcp", region="us-central1"
        )
        assert latest == "lore-mcp-00001-first"
        assert prev is None


@pytest.mark.unit
def test_get_current_and_previous_revisions_empty() -> None:
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["gcloud"], returncode=0, stdout=""
        ),
    ):
        latest, prev = get_current_and_previous_revisions(
            service="lore-mcp", region="us-central1"
        )
        assert latest is None
        assert prev is None


@pytest.mark.unit
def test_get_current_and_previous_revisions_error() -> None:
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1, cmd=["gcloud"], stderr="Service not found"
        ),
    ):
        latest, prev = get_current_and_previous_revisions(
            service="nonexistent", region="us-central1"
        )
        assert latest is None
        assert prev is None


@pytest.mark.unit
def test_get_current_and_previous_revisions_gcloud_not_found() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        latest, prev = get_current_and_previous_revisions(
            service="lore-mcp", region="us-central1"
        )
        assert latest is None
        assert prev is None


@pytest.mark.unit
def test_rollback_traffic_success() -> None:
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["gcloud"], returncode=0, stdout="Traffic routed"
        ),
    ) as mock_run:
        result = rollback_traffic(
            service="lore-mcp",
            region="us-central1",
            target_revision="lore-mcp-00002-xyz",
            project="my-project",
        )
        assert result is True
        cmd = mock_run.call_args[0][0]
        assert "services" in cmd
        assert "update-traffic" in cmd
        assert "--to-revisions=lore-mcp-00002-xyz=100" in cmd
        assert "--project=my-project" in cmd


@pytest.mark.unit
def test_rollback_traffic_failure() -> None:
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1, cmd=["gcloud"], stderr="Permission denied"
        ),
    ):
        result = rollback_traffic(
            service="lore-mcp",
            region="us-central1",
            target_revision="lore-mcp-00002-xyz",
        )
        assert result is False


@pytest.mark.unit
def test_rollback_traffic_file_not_found() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = rollback_traffic(
            service="lore-mcp",
            region="us-central1",
            target_revision="lore-mcp-00002-xyz",
        )
        assert result is False


@pytest.mark.unit
def test_run_rollback_with_explicit_revision() -> None:
    with patch("scripts.rollback.rollback_traffic", return_value=True) as mock_tb:
        exit_code = run_rollback(
            service="lore-mcp",
            region="us-central1",
            target_revision="lore-mcp-00002-prev",
        )
        assert exit_code == 0
        mock_tb.assert_called_once_with(
            service="lore-mcp",
            region="us-central1",
            target_revision="lore-mcp-00002-prev",
            project=None,
        )


@pytest.mark.unit
def test_run_rollback_auto_discovers_previous() -> None:
    with (
        patch(
            "scripts.rollback.get_current_and_previous_revisions",
            return_value=("lore-mcp-00003-broken", "lore-mcp-00002-healthy"),
        ) as mock_get,
        patch("scripts.rollback.rollback_traffic", return_value=True) as mock_tb,
    ):
        exit_code = run_rollback(service="lore-mcp", region="us-central1")
        assert exit_code == 0
        mock_get.assert_called_once_with(
            service="lore-mcp", region="us-central1", project=None
        )
        mock_tb.assert_called_once_with(
            service="lore-mcp",
            region="us-central1",
            target_revision="lore-mcp-00002-healthy",
            project=None,
        )


@pytest.mark.unit
def test_run_rollback_no_previous_found() -> None:
    with patch(
        "scripts.rollback.get_current_and_previous_revisions",
        return_value=("lore-mcp-00001-only", None),
    ):
        exit_code = run_rollback(service="lore-mcp", region="us-central1")
        assert exit_code == 1


@pytest.mark.unit
def test_run_rollback_execution_failure() -> None:
    with (
        patch(
            "scripts.rollback.get_current_and_previous_revisions",
            return_value=("lore-mcp-00003-broken", "lore-mcp-00002-healthy"),
        ),
        patch("scripts.rollback.rollback_traffic", return_value=False),
    ):
        exit_code = run_rollback(service="lore-mcp", region="us-central1")
        assert exit_code == 1


@pytest.mark.unit
def test_deliberately_broken_release_rolls_back() -> None:
    """Verify that a broken release (smoke test fails) triggers rollback."""
    # Simulate post-deploy failure workflow:
    # 1. Broken revision 'rev-broken-002' was deployed.
    # 2. Previous known healthy revision is 'rev-healthy-001'.
    # 3. Rollback executes and returns exit code 0 (successfully rerouting traffic).
    healthy_rev = "lore-mcp-rev-healthy-001"
    broken_rev = "lore-mcp-rev-broken-002"

    with patch("subprocess.run") as mock_subproc:
        # Mock revisions list call
        mock_subproc.side_effect = [
            subprocess.CompletedProcess(
                args=["gcloud", "run", "revisions", "list"],
                returncode=0,
                stdout=f"{broken_rev}\n{healthy_rev}\n",
            ),
            # Mock update-traffic call
            subprocess.CompletedProcess(
                args=["gcloud", "run", "services", "update-traffic"],
                returncode=0,
                stdout=f"Traffic routed to {healthy_rev}=100",
            ),
        ]

        exit_code = run_rollback(service="lore-mcp", region="us-central1")
        assert exit_code == 0
        assert mock_subproc.call_count == 2
        # Check update traffic target
        update_args = mock_subproc.call_args_list[1][0][0]
        assert f"--to-revisions={healthy_rev}=100" in update_args

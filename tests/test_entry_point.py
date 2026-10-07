"""Tests for the entry_point module."""

import contextlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest
import yaml

from ratatosk import io  # , run_modules
from ratatosk.entry_point import entry_point

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_entry_point_help(option: str, capsys) -> None:
    """Test help for the ``entry_point()`` function."""
    with contextlib.suppress(SystemExit):
        entry_point(manually_provided_args=[option])
    output = capsys.readouterr().out
    assert "usage:" in output
    assert "program" in output


@pytest.mark.parametrize(
    (("argument", "option")),
    [
        pytest.param("analysis", "-h", id="process with -h"),
        pytest.param("analysis", "--help", id="process with --help"),
        pytest.param("create-config", "-h", id="create-config with -h"),
        pytest.param("create-config", "--help", id="create-config with --help"),
    ],
)
def test_entry_point_subprocess_help(capsys, argument: str, option: str) -> None:
    """Test the help argument to the sub-entry points."""
    with contextlib.suppress(SystemExit):
        entry_point(manually_provided_args=[argument, option])
    output = capsys.readouterr().out
    assert "usage:" in output
    assert argument in output


@pytest.mark.skip(reason="ns-rse 2026-10-07 Awaiting writing of run_modules module.")
@pytest.mark.parametrize(
    ("options", "expected_function", "expected_args"),
    [
        pytest.param(
            [
                "-c",
                "dummy/config/dir/config.yaml",
                "analysis",
            ],
            # run_modules.analysis,
            "something",
            {"config_file": Path("dummy/config/dir/config.yaml")},
            id="Analysis with config file argument",
        ),
        pytest.param(
            [
                "--output-dir",
                "/tmp/output/",
                "analysis",
            ],
            # run_modules.analysis,
            "something",
            {"output_dir": Path("/tmp/output")},
            id="Analysis with output (long) arguments",
        ),
        pytest.param(
            [
                "-l",
                "debug",
                "--cores",
                "16",
                "analysis",
            ],
            # run_modules.analysis,
            "something",
            {"log_level": "debug", "cores": 16},
            id="Analysis with log_level (short), cores (long)",
        ),
        pytest.param(
            [
                "create-config",
                "-f",
                "dummy_config_file.yaml",
            ],
            io.write_config,
            {"filename": Path("dummy_config_file.yaml")},
            id="Create config with custom filename",
        ),
        pytest.param(
            [
                "create-config",
                "-f",
                "dummy_config_file.yaml",
            ],
            io.write_config,
            {"filename": Path("dummy_config_file.yaml")},
            id="Create config with custom filename (short), custom module (long)",
        ),
    ],
)
def test_entry_points(
    options: list[str], expected_function: Callable, expected_args: dict[str, Any]
) -> None:
    """Ensure the correct function is called for each program, and arguments are carried through correctly."""
    returned_args = entry_point(options, testing=True)
    # convert argparse's Namespace object to dictionary
    returned_args_dict = vars(returned_args)
    # check that the correct function is collected
    assert returned_args.func == expected_function
    # check that the argument has successfully been passed through into the dictionary
    for argument, value in expected_args.items():
        assert returned_args_dict[argument] == value


@pytest.mark.skip(reason="ns-rse 2026-10-07 Awaiting writing of run_modules module.")
@pytest.mark.parametrize(
    ("manual_args"),
    [
        pytest.param(
            [
                "--log-level",
                "info",
                "analysis",
            ],
            id="analysis of XXX",
        ),
    ],
)
def test_analysis(manual_args: list[str], tmp_path: Path, snapshot) -> None:
    """Test for ``run_modules.analysis()``."""
    manual_args = ["--output-dir", str(tmp_path), *manual_args]
    entry_point(manually_provided_args=manual_args)
    # ns-rse 2026-10-07 Update with the expected number of files for a given input
    # Check there are two files in the output directory
    assert sum(1 for _ in tmp_path.iterdir() if _.is_file()) == 2
    # ns-rse 2026-10-07 Add in more checks here, file types and loading for snapshots
    assert snapshot == "a"


# ns-rse 2026-10-07 - End to End test to be modified once we have a working workflow in place
# e2e tests run once per week via github action (yet to be written), skipped in ci.yml
@pytest.mark.e2e
@pytest.mark.skip(reason="ns-rse 2026-10-07 Awaiting writing of run_modules module.")
@pytest.mark.parametrize(
    ("config_file_name"),
    [
        pytest.param(
            "dummy_config.yaml",
            id="test1",
        ),
    ],
)
def test_cli_ratatosk_analysis(
    config_file_name: str,
    tmp_path: Path,
    monkeypatch,
    snapshot,
) -> None:
    """Simulates the CLI and verifies the CSV & png plot output from ratatosk against a snapshot."""
    test_config_path = RESOURCES / "config" / config_file_name

    # Update config and rewrite it to temp file
    with test_config_path.open() as file:
        config = yaml.safe_load(file)
    config["base_dir"] = str(tmp_path)
    tmp_output_path = tmp_path / "output"
    config["output_dir"] = str(tmp_output_path)
    tmp_config_path = tmp_path / config_file_name
    with tmp_config_path.open("w") as file:
        yaml.dump(config, file)

    # Run CLI in-process
    monkeypatch.setattr(
        sys, "argv", ["ratatosk", "-c", str(tmp_config_path), "analysis"]
    )
    entry_point()

    # ns-rse 2026-10-07 - write tests comparing output
    assert snapshot == "a"

"""Test end-to-end running of ratatosk."""

import contextlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from loguru import logger

from ratatosk import RATATOSK_BASE_VERSION, run_module
from ratatosk.entry_point import entry_point

BASE_DIR = Path.cwd()


@pytest.mark.skip(
    reason="Tests pass but are not checking `run_module._set_loggingin()` as can't access a set logging level in loguru"
)
@pytest.mark.parametrize(
    ("log_level", "effective_level"),
    [
        pytest.param("DEBUG", 10, id="log level debug"),
        pytest.param("INFO", 20, id="log level info"),
        pytest.param("WARNING", 30, id="log level warning"),
        pytest.param("ERROR", 40, id="log level error"),
    ],
)
def test_set_logging(log_level: str, effective_level: int) -> None:
    """Test setting log-level."""
    run_module._set_logging(log_level)  # pylint: disable=protected-access
    assert logger.level(name=log_level).no == effective_level


@pytest.mark.parametrize("option", [("-h"), ("--help")])
def test_run_ratatosk_main_help(capsys, option) -> None:
    """Test the -h/--help flag to run_ratatosk."""
    with contextlib.suppress(SystemExit):
        entry_point(manually_provided_args=[option])
    assert "Run Ratatosk analysis." in capsys.readouterr().out


def test_run_ratatosk_process_all(capsys) -> None:
    """Test run_ratatosk completes without error when no arguments are given."""
    # Explicitly force loading of ratatosk/default_config.yaml as I couldn't work out how to invoke process_all()
    # without any arguments as it defaults to 'sys.argv' as this is wrapped within pytest it picks up the arguments
    # pytest was invoked with (see thread on StackOverflow at https://stackoverflow.com/a/55260580/1444043)
    entry_point(
        manually_provided_args=[
            "--config",
            f"{BASE_DIR / 'src' / 'ratatosk' / 'default_config.yaml'}",
            "--base-dir",
            "./tests/resources/xls/",
            "--file-ext",
            "xlsx",
            "process",
        ]
    )
    output = capsys.readouterr().err
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in output
    assert "Files Processed             : 2 (100.0%)" in output


def test_run_ratatosk_process_debug(capsys) -> None:
    """Test run_ratatosk with debugging and check DEBUG messages are logged."""
    entry_point(
        manually_provided_args=[
            "--config",
            f"{BASE_DIR / 'src' / 'ratatosk' / 'default_config.yaml'}",
            "-l",
            "debug",
            "--base-dir",
            "./tests/resources/xls/",
            "--file-ext",
            "xlsx",
            "process",
        ]
    )
    output = capsys.readouterr().err
    assert "File Extension              : xlsx" in output
    assert "Files Processed             : 2 (100.0%)" in output
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in output


@pytest.mark.parametrize(
    ("config_fixture", "input_files", "files_processed", "check"),
    [
        pytest.param(
            "default_parameters",
            5,
            5,
            "File Extension              : xlsx",
            id="dummy output_dir and filename",
        )
    ],
)
def test_completion_message(
    config_fixture: str,
    input_files: int,
    files_processed: int,
    check: str,
    request,
    caplog,
) -> None:
    """Test the `completion_message()` correctly reports parameters."""
    config = request.getfixturevalue(config_fixture)
    run_module.completion_message(
        config, input_files, files_processed, start_time=datetime.now(UTC)
    )
    assert check in caplog.text
    assert "yggdrasil-materials.github.io/ratatosk" in caplog.text
    assert RATATOSK_BASE_VERSION in caplog.text

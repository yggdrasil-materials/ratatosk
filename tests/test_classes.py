"""Tests for the classes module."""

from pathlib import Path

import pytest

from ratatosk import classes

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
YAML = RESOURCES / "yaml"


@pytest.mark.parametrize(
    ("config_file", "base_dir", "output_dir", "log_level", "cores"),
    [
        pytest.param(
            None,
            Path("./"),
            Path("./output"),
            "info",
            2,
            id="Default configuration",
        ),
        pytest.param(
            YAML / "dummy_config.yaml",
            Path("/tmp/"),
            Path("./output"),
            "warning",
            6,
            id="Dummy configuration",
        ),
    ],
)
def test_parameters(
    config_file: Path | None,
    base_dir: Path,
    output_dir: Path,
    log_level: str,
    cores: int,
) -> None:
    """Test for the `Parameters` class."""
    if config_file is None:
        parameters = classes.Parameters()
    else:
        parameters = classes.Parameters(config_file=config_file)
    assert parameters.base_dir == base_dir
    assert parameters.output_dir == output_dir
    assert parameters.log_level == log_level
    assert parameters.cores == cores

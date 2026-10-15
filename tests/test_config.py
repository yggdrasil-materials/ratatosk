"""Test the config module function(s)."""

import argparse
import logging
from pathlib import Path
from pkgutil import get_data

import pytest
import yaml

from ratatosk import config

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
YAML_DIR = RESOURCES / "yaml"

default_config = get_data(package="ratatosk", resource="default_config.yaml")
DEFAULT_CONFIG = yaml.full_load(default_config)


def test_reconcile_config_args_no_config() -> None:
    """Test the handling of config file function with no config."""
    args = argparse.Namespace(
        program="process",
        config_file=None,
        output_dir="dummy_dir",
    )
    _config = config.reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)
    assert _config["output_dir"] == Path("./dummy_dir")


def test_reconcile_config_args_no_config_with_overrides() -> None:
    """Test the handle config file function with no config and overrides."""
    args = argparse.Namespace(
        program="process",
        config_file=None,
        base_dir="./somewhere/else",
        output_dir="./dummy_output_dir",
    )
    _config = config.reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)
    # Check that the overrides have been applied
    assert _config["base_dir"] == Path("./somewhere/else")
    assert _config["output_dir"] == Path("./dummy_output_dir")


def test_reconcile_config_args_full_config() -> None:
    """Test the handle config file function with a full config."""
    args = argparse.Namespace(
        program="process",
        config_file=f"{YAML_DIR / 'dummy_config.yaml'}",
    )
    _config = config.reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)
    assert _config["base_dir"] == Path("/tmp/")
    assert _config["output_dir"] == Path("./output")
    assert _config["cores"] == 6
    assert _config["log_level"] == "warning"


def test_reconcile_config_args_partial_config() -> None:
    """Test the reconcile_config_args function with a partial config."""
    args = argparse.Namespace(
        program="process",
        config_file=f"{YAML_DIR / 'partial_config.yaml'}",
    )
    _config = config.reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)
    # Check that the partial config has overridden the default config
    assert _config["cores"] == 6
    assert _config["log_level"] == "warning"


def test_reconcile_config_args_partial_config_with_overrides() -> None:
    """Test the reconcile_config_args function with a partial config and overrides."""
    args = argparse.Namespace(
        program="process",
        config_file=f"{YAML_DIR / 'partial_config.yaml'}",
        output_dir="./dummy_output_dir",
    )
    _config = config.reconcile_config_args(args=args, default_config=DEFAULT_CONFIG)

    # Check that the partial config has overridden the default config
    assert _config["cores"] == 6
    assert _config["log_level"] == "warning"
    # Check that the overrides have been applied
    assert _config["output_dir"] == Path("./dummy_output_dir")


@pytest.mark.parametrize(
    ("dict1", "dict2", "expected_merged_dict"),
    [
        pytest.param(
            {"a": 1, "b": 2},
            {"c": 3, "d": 4},
            {"a": 1, "b": 2, "c": 3, "d": 4},
            id="two dicts, no common keys",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
            {"a": 1, "b": 3, "c": 4},
            id="two dicts, one common key, testing priority of second dict",
        ),
        # Nested dictionaries
        pytest.param(
            {"a": 1, "b": {"c": 2, "d": 3}},
            {"b": {"c": 4, "e": 5}},
            {"a": 1, "b": {"c": 4, "d": 3, "e": 5}},
            id="nested dictionaries, one common key in nested dict, testing priority of second dict",
        ),
    ],
)
def test_merge_mappings(dict1: dict, dict2: dict, expected_merged_dict: dict) -> None:
    """Test merging of mappings."""
    merged_dict = config.merge_mappings(dict1, dict2)
    assert merged_dict == expected_merged_dict


@pytest.mark.parametrize(
    ("filename", "config_arg", "expected_filename"),
    [
        pytest.param(
            "test_config_with_comments.yaml",
            None,
            "test_config_with_comments.yaml",
            id="filename, no config name",
        ),
        pytest.param(
            "test_config_with_comments",
            None,
            "test_config_with_comments.yaml",
            id="filename without yaml extension, no config name",
        ),
        pytest.param(None, None, "config.yaml", id="no filename, no config name"),
    ],
)
def test_write_config_with_comments(
    tmp_path: Path, filename: str, config_arg: str, expected_filename: str
) -> None:
    """Test `config.write_config_with_comments()`."""
    # Setup argparse.Namespace with the tests parameters
    args = argparse.Namespace()
    args.filename = filename
    args.output_dir = tmp_path
    args.config = config_arg
    args.module = "ratatosk"
    # Write default config with comments to file
    config.write_config_with_comments(args)
    assert Path(tmp_path / expected_filename).is_file()
    # Read the written config
    with Path.open(tmp_path / expected_filename, encoding="utf-8") as f:
        written_config = f.read()
    # Validate that the written config has comments in it
    print(f"\n{written_config=}\n")
    assert "Config file generated" in written_config
    assert "For more information on configuration and how to use it" in written_config
    # Validate some of the parameters are present conditional on config type
    assert "base_dir: ./" in written_config
    assert "output_dir: ./output" in written_config
    assert "cores: 2" in written_config


@pytest.mark.parametrize(
    ("module", "config_str", "error"),
    [
        pytest.param(
            "ratatosk",
            "nonsense",
            ValueError,
            id="rataosk module for 'nonsense' config",
        ),
        pytest.param(
            "other_package",
            "default",
            ValueError,
            id="other_package module for 'default' config",
        ),
    ],
)
def test_write_config_with_comments_user_warning(
    module: str, config_str: str, error, tmp_path: Path
) -> None:
    """Tests if errors are raised when an attempt is made to request a configuration file type that does not exist."""
    args = argparse.Namespace()
    args.filename = "config.yaml"
    args.output_dir = tmp_path
    args.config = config_str
    args.module = module
    with pytest.raises(error):
        config.write_config_with_comments(args)


def test_update_config(caplog) -> None:
    """Test updating configuration."""
    caplog.set_level(logging.DEBUG)
    sample_config = {
        "a": 1,
        "b": 2,
        "c": "something",
        "base_dir": "here",
        "output_dir": "there",
    }
    new_values = {"c": "something new"}
    updated_config = config.update_config(config=sample_config, args=new_values)

    assert isinstance(updated_config, dict)
    assert "Updated config config[c] : something > something new" in caplog.text
    assert updated_config["c"] == "something new"

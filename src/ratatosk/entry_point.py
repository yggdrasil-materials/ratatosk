"""Argument parsing and entry point for running rataosk at the command line."""

from __future__ import annotations

import argparse as arg
import sys
from pathlib import Path
from typing import Any

from ratatosk import __version__  # , run_modules
from ratatosk.io import write_config


def ratatosk_parser() -> arg.ArgumentParser:
    """
    Create a parser for reading options at the commandline.

    The parser has multiple sub-parsers for reading options to run ``layopt``

    Returns
    -------
    arg.ArgumentParser
        Argument parser.
    """
    parser = arg.ArgumentParser(description="Run layopt.")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Installed version of Layopt: {__version__}",
        help="Report the current version of Layopt that is installed",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        type=Path,
        required=False,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=False,
        help="Output directory to write results to.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str,
        required=False,
        help="Set verbosity of logging, options (least verbose to most) are 'error', 'warning', 'info', 'error', 'debug'.",
    )
    parser.add_argument(
        "-j",
        "--cores",
        dest="cores",
        type=int,
        required=False,
        help="Number of cores to use for parallel processing.",
    )
    # Add subparsers
    subparsers = parser.add_subparsers(
        title="program",
        description="Available processing options are :",
        dest="module",
    )

    # Add an analysis parser
    analysis_parser = subparsers.add_parser(
        "analysis",
        description="Run Ratatosk analysis",
        help="Run Rataosk analysis",
    )
    # ns-rse 2026-10-07 place holder, yet to write run_modules
    analysis_parser.set_defaults()

    # Add a create configuration parser
    create_config_parser = subparsers.add_parser(
        "create-config",
        description="Create a configuration file using the defaults.",
        help="Create a configuration file using the defaults.",
    )
    create_config_parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        type=Path,
        required=False,
        help="Name of YAML file to save configuration to (default 'config.yaml').",
    )
    create_config_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=False,
        help="Path to where the YAML file should be saved (default './' the current directory).",
    )
    create_config_parser.add_argument(
        "-t",
        "--type",
        dest="type",
        type=str,
        default="config",
        help="Configuration to write, default is 'config', other option is 'data-dictionary'.",
    )
    create_config_parser.set_defaults(func=write_config)

    return parser


def entry_point(
    manually_provided_args: list[Any] | None = None, testing: bool = False
) -> arg.Namespace | None:
    """
    Entry point for all Ratatosk programs.

    Main entry point for running ``ratatosk`` which allows the different processing, plotting and testing modules to be
    run.

    Parameters
    ----------
    manually_provided_args : None
        Manually provided arguments.
    testing : bool
        Whether testing is being carried out.

    Returns
    -------
    None
        Function does not return anything.
    """
    # Create Ratatosk parser
    parser = ratatosk_parser()
    args = (
        parser.parse_args()
        if manually_provided_args is None
        else parser.parse_args(manually_provided_args)
    )
    # If no module has been specified print help and exit
    if not args.module:
        parser.print_help()
        sys.exit()
    if testing:
        return args
    # Run the specified module(s)
    args.func(args)
    return None

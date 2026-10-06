"""Dataclass definitions."""

from pathlib import Path

# import numpy as np
# import numpy.typing as npt
# from loguru import logger
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(
    repr=True,
    eq=True,
    config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True),
    validate_on_init=True,
)
class Parameters:
    """Parameters."""

    base_dir: Path = Field(default=Path("./"), title="Base directory")
    output_dir: str | Path = Field(
        default=Path("./output/"),
        title="Path to save the output to, default is './output/'.",
    )
    log_level: str = Field(default="info", title="Log level")
    cores: int = Field(default=2, title="Cores to run optimisation on in parallel.")

"""Dataclass definitions."""

from pathlib import Path
from typing import Any

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


@dataclass(
    repr=True,
    eq=True,
    config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True),
    validate_on_init=True,
)
class DataDictionary:
    """Data Dictionary."""

    neware: dict[str, Any] = Field(
        default={
            "columns": {
                "Cycle Index": "Cycle",
                "Step Type": "Step",
                "Voltage(V)": "Voltage",
                "Chg. Spec. Cap.(mAh/g)": "Charge_Capacity",
                "DChg. Spec. Cap.(mAh/g)": "Discharge_Capacity",
                "dQm/dV(mAh/V.g)": "dQ/dV",
            },
            "numeric": [
                "Cycle",
                "Voltage",
                "Charge_Capacity",
                "Discharge_Capacity",
                "dQ/dV",
            ],
            # ns-rse 2026-10-07 - Can likely do away with this and use regex to determine charge/discharge, based on below the
            # presence of 'D' indicates "Discharge", otherwise state is "Charge" (do it case-insensitive just to be
            # sure). If we can't though then we should flip this round, have 'Charge' and 'Discharge' as keys and the
            # possible values a set that can be compared against
            "step_names": {
                "CCCV Chg": "Charge",
                "CCCV CHG": "Charge",
                "CC Chg": "Charge",
                "CC CHG": "Charge",
                "CC DChg": "Discharge",
                "CC DCHG": "Discharge",
                "CCCV DChg": "Discharge",
            },
        },
        title="Neware data structure",
    )

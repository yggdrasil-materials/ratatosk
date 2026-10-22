# Classes

As well as the [API](../api/index.md) reference for these classes further details are given below on the various classes
implemented in Ratatosk.

## Dataclasses

Details of the [Pydantic Dataclasses][pyd_dataclass] and their attributes are given below.

### `Parameters`

The `Parameters` dataclass holds the configuration options for undertaking the analysis. These are loaded from the
packages included YAML file [`default_config.yaml`][dc].

| Attribute                    | Type    | Default      | Description                                                        |
|------------------------------|---------|--------------|--------------------------------------------------------------------|
| `base_dir`                   | `Path`  | `./`         | Base directory where code is being run and files are searched for  |
| `output_dir`                 | `Path`  | `./output`   | Where to save output to                                            |
| `log_level`                  | `str`   | `info`       | Verbosity used in logging                                          |
| `cores`                      | `int`   | `2`          | Number of cores to use when processing                             |
| `file_ext`                   | `str`   | `xlsx`       | File extension of results to analyse                               |
| `terminal_trim`              | `bool`  | `True`       | Whether to perform trimming of half-cycles                         |
| `terminal_trim_fixed_mv`     | `float` | `75.0`       | ???                                                                |
| `terminal_trim_factor`       | `float` | `1.4`        | ???                                                                |
| `terminal_trim_max_fraction` | `float` | `.1`         | ???                                                                |
| `terminal_trim_band_mv`      | `float` | `20.0`       | ???                                                                |
| `terminal_trim_interior`     | `float` | `[0.2, 0.8]` | ???                                                                |
| `dqdv_computation`           | `str`   | `histogram`  | Differential method, valid values are `histogram` or `derivative`  |

### `FileStructure`

The `FileStructure` dataclass contains details on the expected fields and values of different input file types and is
used to make extending support for different input filetypes. Each attribute is a dictionary of parameters associated
with a file type or key component of analysis. The keys and values of the dictionary are explained. These are loaded from the
packages included YAML file [`default_dictionary.yaml`][dd].

| Attribute | Type             | Keys         | Values           | Description                                                                    |
|-----------|------------------|--------------|------------------|--------------------------------------------------------------------------------|
| `neware`  | `dict[str, Any]` | `columns`    | `dict[str, str]` | Mapping of column names as read to desired column names.                         |
|           |                  | `numeric`    | `list[str]`      | List of columns that _must_ be numerical.                            |
|           |                  | `step_names` | `dict[str, str]` | Dictionary mapping possible abbreviation variations to `Charge` / `Discharge`. |

[dc]: https://github.com/yggdrasil-materials/ratatosk/bob/main/src/ratatosk/default_config.yaml
[dd]: https://github.com/yggdrasil-materials/ratatosk/bob/main/src/ratatosk/default_dictionary.yaml
[pyd_dataclass]: https://pydantic.dev/docs/validation/latest/concepts/dataclasses/

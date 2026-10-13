# Usage

Once you have [installed](install.md) `ratatosk` in a virtual environment you are ready to start using it. There are two
methods, the Command Line Interface (CLI) or via Notebooks (either [Jupyter][jupyter] or [Marimo][marimo])

## Command Line Interface (CLI)

Once you have [installed](install.md) `ratatosk` you are ready to start using it. The command-line interface (CLI) has
built-in help available for the main command `ratatosk --help`...

```shell
❱ ratatosk --help
usage: ratatosk [-h] [-v] [-c CONFIG_FILE] [-o OUTPUT_DIR] [-l LOG_LEVEL] [-j CORES] {analysis,create-config} ...

Run layopt.

options:
  -h, --help            show this help message and exit
  -v, --version         Report the current version of Layopt that is installed
  -c, --config-file CONFIG_FILE
                        Path to a YAML configuration file.
  -o, --output-dir OUTPUT_DIR
                        Output directory to write results to.
  -l, --log-level LOG_LEVEL
                        Set verbosity of logging, options (least verbose to most) are 'error', 'warning', 'info', 'error', 'debug'.
  -j, --cores CORES     Number of cores to use for parallel processing.

program:
  Available processing options are :

  {analysis,create-config}
    analysis            Run Rataosk analysis
    create-config       Create a configuration file using the defaults.
```

...and programs (e.g. `ratatosk create-config --help` or `ratatosk analysis --help`).

### Configuration File

The first step is to create your own configuration file which can be done using `ratatosk create-config`. This will
create a [YAML][yaml] configuration file that you can edit in a text editor[^1].By default it will write to
`default_config.yaml` but you can modify the output filename using the `-f` / `--filename` potion. There are two
possible configuration files that can be written either `--type config` or `--type default-dictionary`. Typically you
will only ever want to use the former as it the dictionary is only for adding support for new file types. If you do not
specify a `--type` option it defaults to writing a configuration file rather than dictionary so safely omit the `--type`
option.

```shell
(.venv) ❱ ratatosk create-config  -f custom_config.yaml
2026-10-12 16:06:52.569 | INFO     | ratatosk.io:write_config:209 - A sample configuration has been written to : custom_config.yaml
```

The resulting `custom_config.yaml` is fully commented indicating what each configuration option does.

**TODO** expand this once we have `ratatosk analysis` working and all options in place.

```yaml
# Config generated 2026-10-12 16:06:52
# For more information on configuration and how to use it:
# https://yggrdrasil-materials.github.io/ratatosk/
# Ratatosk version : 0.1.0
# Commit:
base_dir: .
output_dir: output
log_level: info
cores: 2
config_file:
  /home/neil/work/git/hub/yggdrasil-materials/ratatosk/ns-rse/70-contributing-docs/src/ratatosk/default_config.yaml
```

Once you have change the parameter(s) you wish to modify you should save the file and return to the CLI.

### Running `ratatosk`

**TODO** write this once we have `ratatosk analysis` working.

## Notebooks

**NB** Whilst you can use either [Jupyter][jupyter] or [Marimo][marimo] notebooks it is recommended to use the later as
they solve one of the major problems of [cell dependency][marimo_reactive] and the problems that arise when these are
run out of order.

[jupyter]: https://jupyter.org/
[marimo]: https://marimo.io
[marimo_reactive]: https://docs.marimo.io/guides/reactivity/
[yaml]: https://yaml.org

[^1]: **NB** Microsoft word is _not_ a text editor and should not be used to edit YAML files. You can use `nano`, `vim`
    or `emacs` on GNU/Linux and OSX systems or `notepad`, `emacs` or your choice of IDE on Windows.

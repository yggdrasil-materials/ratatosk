# Installation

There are several different methods for installing `ratatosk` on your system. In all cases it is recommended that you
use a Python Virtual Environment and install `ratatosk` within that environment. Development uses [uv] and the
instructions below therefore show how to setup virtual environments using `uv`.

If you are considering contributing to the development of `ratatosk` you will find more detailed instructions on the
[contributing](contributing.md) page.

## GitHub installation

There are two methods to install the package from GitHub described below.

### Install with `pip`

You can use the `pip` tool to install directly from GitHub. The following will install the `main` branch, but you can
specify any branch, tag or commit by changing what follows the `.git@` at the end from `main` to the branch or tag you
wish to install (or even specific commit hash).

```
pip install @ git+https://github.com/yggdrasil-materials/ratatosk.git@main
```

### Cloning and installing

You can clone the repository, create a virtual environment within the cloned directory and install `ratatosk` from
there. This is described in detail in the [contributing](contributing.md) section of the site.

## Python Package Index (PyPI) install

**NB** `ratatosk` has not yet been released to PyPI so the following commands will not work. Please use the GitHub
method described above.

Create a directory in which you will create a virtual environment.

=== "GNU/Linux or OSX"

```shell
mkdir -p ~/work/ratatosk
cd ~/work/ratatosk
uv venv --python 3.14
source .venv/bin/activate
uv sync
uv pip install ratatosk
```

=== "Windows"

```shell
mkdir -p ~/work/ratatosk
cd ~/work/ratatosk
uv venv --python 3.14
source .venv\Scripts\activate
uv sync
uv pip install ratatosk
```

## Checking installation

Once installed you should be able to run `ratatosk --help` and see output as shown below describing the command line
options.

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

Each of the sub-commands also have help options available which can be viewed with e.g. `ratatosk create-config --help`.

For more details on using and running `ratatosk` see the [usage](usage.md) pages.

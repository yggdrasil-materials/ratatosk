# Contributing

If you have a feature or improvement you would like to make to Ratatosk you can do so by forking the repository and
making a pull request (members of the `yggdrasil-materials` organisation should have access to clone and work
on/contribute to the repository directly).

We ask that you read and abide by our [Code of Conduct][cc] which is based on the [Contributors Covenant][cc].

We have adopted various style guides and use a number of [pre-commit][pc] hooks to ensure code is formatted and linted
consistently which helps make it easier for developers to read and understand each others contributions.

## Clone the Repository

If you are a member of the [yggdrasil-materials][ym] organisation or have been invited to work on the [ratatosk
repository][ratatosk] you should be able to clone and work with the repository directly. If not contributions are still
very welcome but you will have to [fork][gh_fork] the repository and make pull requests from your fork.

## Development Setup

Ratatosk is developed using [uv][uv] to manage virtual environments, package dependencies and semantic versioning based
on Git Tags. In order to get started with contributing you should [install uv][uv_install] and create a Python Virtual
Environment using Python 3.14.

```shell
uv venv --python 3.14
```

After creating a virtual environment you will _always_ need to activate it so that the `$PATH` variable has the path to
the virtual environment prepended.  Depending on your operating system you will have to do one of the following to
activate it:

<!-- markdownlint-disable MD046 -->
=== "GNU/Linux or OSX"

    ``` shell
    .venv/bin/activate
    ```

=== "Windows"

    ``` shell
    .venv\Scripts\activate
    ```
<!-- markdownlint-enable MD046 -->

<!-- markdownlint-disable MD046 -->
!!! tip

    GNU/Linux and OSX users can automate this using the [`direnv`][direnv]
    programme and adding the following to the `.envrc` file.

    ``` shell
    source .venv/bin/activate
    export PATH=${VIRTUAL_ENV}/bin:${PATH}
    ```

    You will have to run `direnv allow` in the cloned repository to allow
    `direnv` to use this script but from that point on when you `cd` into the
    directory the environment should be activated.
<!-- markdownlint-enable MD046 -->

### Install the package for development

You can install all package dependencies with `uv sync` and ensure all development dependencies are installed using the
`--group dev` flag to `uv pip install`

```shell
uv sync
uv pip install -e . --group dev
```

Once installed you should be able to run the following commands and see the output similar to that shown below, although
note that the exact version may differ slightly on your install.

#### `python --version`

``` shell
❱ python --version
Python 3.14.8
```

#### `pre-commit --version`

``` shell
❱ pre-commit --version
```

#### `ratatosk --help`

``` shell
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

## Style

We aim to adhere to the widely used [PEP8][pep8] style guide for writing code. This can be applied using tools such as
[`black`][black] and/or [`ruff`][ruff] and [`pylint`][pylint].

## Type-hints

Ratatosk uses [Pydantic][pydantic] to help with strong typing of variables. This helps ensure the right data types are
passed between classes and functions/methods and reduces the scope for unexpected errors. To help with this we use both
[mypy][mypy] and the newer (faster) [ty][ty] static type-checkers as pre-commit hooks.

You may find your Integrated Development Environment (IDE) uses both the `ty` built-in language server and those of the
linter [`ruff`][ruff] if you have them installed and available. Because of the scope for variation in the IDEs that are
used though details on how to install and configure are not covered in this documentation. Please refer to your IDEs
documentation or that of [ty][ty_lsp] and [ruff][ruff_lsp]

## Pre-commit

The [Pre-commit][pc] framework provides a system for running [Git hooks][git_hooks] prior to making commits. These
ensure that code is linted and formatted consistently according to [PEP8][pep8] guidelines which makes it easier for
those reviewing and subsequently working with your code to understand it.

Once you have setup your virtual environment you can install the pre-commit hooks using `pre-commit install`

```shell
pre-commit install
```

For details of all pre-commit hooks that are used in this project please see the `.pre-commit-config.yaml` file at the
root of the cloned repository which lists the `repo` and each of the `hooks` that are configured.

## Git Conventions

When making branches we ask that you use informative branch names that indicate the person working on the branch and the
issue they are working on. For example the user with GitHub account name `ns-rse` working on issue [70 Document how to
contribute and add Code of Conduct](https://github.com/yggdrasil-materials/ratatosk/issues/70) create a branch
`ns-rse/70-contributing-docs` to undertake the work on. This follows the nomenclature of using the GitHub username
(`<github_username>`) followed by a `/`, then the `<issue_number>` and a `-short-description`. This makes it quick and
easy for others, including your future self, to see who the branch belongs to and the issue that it is addressing.

### Pre-commit usage

Once `pre-commit` has been installed and configured you should find the hooks that check the code based are run on all
files included in a given commit, but _not_ those files unaffected by a given commit.

If errors are reported we would appreciate if you resolve them locally first _before_ making pull requests to avoid
running the same hooks on Continuous Integration on GitHub where they will invariably fail. By fixing issues _before_
commits are made it helps reduce cloud computing resources which reduces environmental impact and has the added benefit
that commit history will be a little cleaner.

<!-- markdownlint-disable MD046 -->
!!! warning "GitKraken Users"

    If you use [GitKraken][gk] to manage your Git workflow you may well find that the hooks are _not_ run when making
    commits. You may find this can be solved by going to _File > Preferences > Hooks_ (this is under the _Repo-specific
    preferences on the right hand side and may require scrolling down_). You can then explicitly set the directory that
    hooks are run from to the `.git/hooks` sub-directory of your cloned repository. If this doesn't work it may be
    because the Virtual Environment, in the examples here managed by `uv`, is not active and so the `pre-commit` command
    can not be found. In which case you should do all your staging in GitKraken and then use a Command Line Interface
    (CLI) via [Git Bash for Windows][gb] to make your commit
<!-- markdownlint-enable MD046 -->

<!-- markdownlint-disable MD046 -->
!!! tip "Git Bash"

    When installing [Git Bash for Windows][gb] you may be asked what editor to use for editing commit messages. It is
    recommended to select `nano` as the default (its included when installing Git Bash), you can the use `git commit -e`
    to edit your commit message. When done use `Ctrl + x` to exit and answer `Yes` when prompted to save changes.
<!-- markdownlint-enable MD046 -->

<!-- markdownlint-disable -->
[black]: https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html
[cc]: https://www.contributor-covenant.org/
[gh_fork]: https://docs.github.com/articles/fork-a-repo
[git_hooks]: https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
[mypy]: https://mypy-lang.org/
[pc]: https://pre-commit.com
[pep8]: https://peps.python.org/pep-0008/
[pydantic]: https://pydantic.dev/docs/
[pylint]: https://pylint.pycqa.org/en/latest/index.html
[ratatosk]: https://github.com/yggdrasil-materials/ratatosk/
[ruff]: https://docs.astral.sh/ruff
[ruff_lsp]: https://docs.astral.sh/ruff/editors/
[ty]: https://docs.astral.sh/ty
[ty_lsp]: https://docs.astral.sh/ty/features/language-server
[uv]: https://docs.astral.sh/uv
[uv_install]: https://docs.astral.sh/uv/getting-started/installation/
[ym]: https://github.com/yggdrasil-materials/
<!-- markdownlint-enable -->

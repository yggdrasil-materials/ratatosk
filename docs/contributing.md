# Contributing

If you have a feature or improvement you would like to make to Ratatosk you can
do so by forking the repository and making a pull request.

We use a number of [pre-commit][pc] hooks to ensure code is formatted and linted
consistently which helps make it easier for developers to understand each others
contributions.

## Clone the Repository

If you are a member of the [yggdrasil-materials][ym] organisation or have been
invited to work on the [ratatosk repository][ratatosk] you should be able to
clone and work with the repository directly. If not contributions are still very
welcome but you will have to [fork][gh_fork] the repository and make pull
requests from your fork.

## Development Setup

Ratatosk is developed using [uv][uv] to manage virtual environments, package
dependencies and semantic versioning based on Git Tags. In order to get started
with contributing you should [install uv][uv_install] and create a Python
Virtual Environment using Python 3.14.

```shell
uv venv --python 3.14
```

You can install all package dependencies with `uv sync` and ensure all
development dependencies are installed using the `--group dev` flag to
`uv pip install`

```shell
uv sync
uv pip install -e . --group dev
```

## Pre-commit

The [Pre-commit][pc] framework provides a system for running [Git
hooks][git_hooks] prior to making commits. These ensure that code is linted and
formatted consistently according to [PEP8][pep8] guidelines which makes it
easier for those reviewing and subsequently working with your code to understand
it.

Once you have setup your virtual environment you can install the pre-commit
hooks using `pre-commit install`

```shell
pre-commit install
```

For details of all pre-commit hooks that are used in this project please see the
`.pre-commit-config.yaml` file at the root of the cloned repository.

### Type-hints

Ratatosk uses [Pydantic][pydantic] to help with strong typing of variables. This
helps ensure the right data types are passed between classes and
functions/methods and reduces the scope for unexpected errors. To help with this
we use both [mypy][mypy] and the newer (faster) [ty][ty] static type-checkers as
pre-commit hooks.

You may find your Integrated Development Environment (IDE) uses both the `ty`
built-in language server and those of the linter [`ruff`][ruff] if you have them
installed and available. Because of the scope for variation in the IDEs that are
used though details on how to install and configure are not covered in this
documentation. Please refer to your IDEs documentation or that of [ty][ty_lsp]
and [ruff][ruff_lsp]

[gh_fork]: https://docs.github.com/articles/fork-a-repo
[git_hooks]: https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
[mypy]: https://mypy-lang.org/
[pc]: https://pre-commit.com
[pep8]: https://peps.python.org/pep-0008/
[pydantic]: https://pydantic.dev/docs/
[ratatosk]: https://github.com/yggdrasil-materials/ratatosk/
[ruff]: https://docs.astral.sh/ruff
[ruff_lsp]: https://docs.astral.sh/ruff/editors/
[ty]: https://docs.astral.sh/ty
[ty_lsp]: https://docs.astral.sh/ty/features/language-server
[uv]: https://docs.astral.sh/uv
[uv_install]: https://docs.astral.sh/uv/getting-started/installation/
[ym]: https://github.com/yggdrasil-materials/

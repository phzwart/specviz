# specviz

<div align="center">

[![Build status](https://github.com/specviz/specviz/workflows/build/badge.svg?branch=master&event=push)](https://github.com/specviz/specviz/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/specviz.svg)](https://pypi.org/project/specviz/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/specviz/specviz/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/specviz/specviz/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/specviz/specviz/releases)
[![License](https://img.shields.io/github/license/specviz/specviz)](https://github.com/specviz/specviz/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

viz of spectra

</div>

## About

SpecViz is a comprehensive spectral visualization and analysis toolkit designed for hyperspectral data processing, machine learning, and interactive exploration. It provides a suite of tools for:

- **Spectral Data Processing**: Load, normalize, and preprocess hyperspectral data
- **Dimensionality Reduction**: UMAP-based visualization and exploration
- **Machine Learning**: Ensemble classifiers, nearest neighbor methods, and conformal prediction
- **Interactive Visualization**: Web-based dashboards for data exploration
- **Database Management**: Efficient storage and retrieval using DuckDB and Redis
- **Parallel Computing**: Distributed processing with Dask

## Key Features

- **Interactive Web Interfaces**: Built with Dash and Plotly for real-time data exploration
- **Machine Learning Pipeline**: Support for various classifiers including RFF, KNN, and ensemble methods
- **Spatial Analysis**: Tools for spatial latent space exploration and trajectory analysis
- **Data Interpolation**: Multiple interpolation methods for spatial data processing
- **Conformal Prediction**: Uncertainty quantification for machine learning predictions
- **Parallel Processing**: Efficient handling of large datasets with Dask
- **Database Integration**: Seamless integration with DuckDB and Redis for data persistence
- **Tools Package**: Local utilities for database operations and other helper functions

## Very first steps

### Initialize your code

1. Initialize `git` inside your repo:

```bash
cd specviz && git init
```

2. If you don't have `Poetry` installed run:

```bash
make poetry-download
```

3. Initialize poetry and install `pre-commit` hooks:

```bash
make install
make pre-commit-install
```

4. Run the codestyle:

```bash
make codestyle
```

5. Upload initial code to GitHub:

```bash
git add .
git commit -m ":tada: Initial commit"
git branch -M main
git remote add origin https://github.com/specviz/specviz.git
git push -u origin main
```

### Set up bots

- Set up [Dependabot](https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates) to ensure you have the latest dependencies.
- Set up [Stale bot](https://github.com/apps/stale) for automatic issue closing.

### Poetry

Want to know more about Poetry? Check [its documentation](https://python-poetry.org/docs/).

<details>
<summary>Details about Poetry</summary>
<p>

Poetry's [commands](https://python-poetry.org/docs/cli/#commands) are very intuitive and easy to learn, like:

- `poetry add numpy@latest`
- `poetry run pytest`
- `poetry publish --build`

etc
</p>
</details>

### Building and releasing your package

Building a new version of the application contains steps:

- Bump the version of your package `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions](https://semver.org/) standard.
- Make a commit to `GitHub`.
- Create a `GitHub release`.
- And... publish 🙂 `poetry publish --build`

## 🎯 What's next

Well, that's up to you 💪🏻. I can only recommend the packages and articles that helped me.

- [`Typer`](https://github.com/tiangolo/typer) is great for creating CLI applications.
- [`Rich`](https://github.com/willmcgugan/rich) makes it easy to add beautiful formatting in the terminal.
- [`Pydantic`](https://github.com/samuelcolvin/pydantic/) – data validation and settings management using Python type hinting.
- [`Loguru`](https://github.com/Delgan/loguru) makes logging (stupidly) simple.
- [`tqdm`](https://github.com/tqdm/tqdm) – fast, extensible progress bar for Python and CLI.
- [`IceCream`](https://github.com/gruns/icecream) is a little library for sweet and creamy debugging.
- [`orjson`](https://github.com/ijl/orjson) – ultra fast JSON parsing library.
- [`Returns`](https://github.com/dry-python/returns) makes you function's output meaningful, typed, and safe!
- [`Hydra`](https://github.com/facebookresearch/hydra) is a framework for elegantly configuring complex applications.
- [`FastAPI`](https://github.com/tiangolo/fastapi) is a type-driven asynchronous web framework.

Articles:

- [Open Source Guides](https://opensource.guide/).
- [A handy guide to financial support for open source](https://github.com/nayafia/lemonade-stand)
- [GitHub Actions Documentation](https://help.github.com/en/actions).
- Maybe you would like to add [gitmoji](https://gitmoji.carloscuesta.me/) to commit names. This is really funny. 😄

## 🚀 Features

### Development features

- Supports for `Python 3.9` and higher.
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/specviz/specviz/blob/master/pyproject.toml) and [`setup.cfg`](https://github.com/specviz/specviz/blob/master/setup.cfg).
- Automatic codestyle with [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade).
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with code-formatting.
- Type checks with [`mypy`](https://mypy.readthedocs.io); docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint); security checks with [`safety`](https://github.com/pyupio/safety) and [`bandit`](https://github.com/PyCQA/bandit)
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Ready-to-use [`.editorconfig`](https://github.com/specviz/specviz/blob/master/.editorconfig), [`.dockerignore`](https://github.com/specviz/specviz/blob/master/.dockerignore), and [`.gitignore`](https://github.com/specviz/specviz/blob/master/.gitignore). You don't have to worry about those things.

### Deployment features

- `GitHub` integration: issue and pr templates.
- `Github Actions` with predefined [build workflow](https://github.com/specviz/specviz/blob/master/.github/workflows/build.yml) as the default CI/CD.
- Everything is already set up for security checks, codestyle checks, code formatting, testing, linting, docker builds, etc with [`Makefile`](https://github.com/specviz/specviz/blob/master/Makefile#L89). More details in [makefile-usage](#makefile-usage).
- [Dockerfile](https://github.com/specviz/specviz/blob/master/docker/Dockerfile) for your package.
- Always up-to-date dependencies with [`@dependabot`](https://dependabot.com/). You will only [enable it](https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates).
- Automatic drafts of new releases with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). You may see the list of labels in [`release-drafter.yml`](https://github.com/specviz/specviz/blob/master/.github/release-drafter.yml). Works perfectly with [Semantic Versions](https://semver.org/) specification.

### Open source community features

- Ready-to-use [Pull Requests templates](https://github.com/specviz/specviz/blob/master/.github/PULL_REQUEST_TEMPLATE.md) and several [Issue templates](https://github.com/specviz/specviz/tree/master/.github/ISSUE_TEMPLATE).
- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.
- [`Stale bot`](https://github.com/apps/stale) that closes abandoned issues after a period of inactivity. (You will only [need to setup free plan](https://github.com/marketplace/stale)). Configuration is [here](https://github.com/specviz/specviz/blob/master/.github/.stale.yml).
- [Semantic Versions](https://semver.org/) specification with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter).

## Installation

### Prerequisites

This project requires Python 3.9 or higher and several system dependencies:

- **Redis**: For data storage and communication between components
- **DuckDB**: For local database operations (included as Python package)
- **Hiposa**: For sampling and workflow tools (PoissonTiler, etc.)

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/specviz/specviz.git
cd specviz
```

2. Install dependencies using Poetry (recommended):
```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

3. Or install using pip:
```bash
pip install -r requirements.txt
```

### Development Installation

For development, install with all development dependencies:

```bash
# Using Poetry
poetry install --with dev

# Using pip
pip install -r requirements.txt
pip install -e .
```

### Running the Application

After installation, you can run:

```bash
# Using Poetry
poetry run specviz --help

# Using pip
specviz --help
```

### System Dependencies

Make sure you have Redis running:

```bash
# On Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# On macOS with Homebrew
brew install redis
brew services start redis

# On Windows
# Download and install Redis from https://redis.io/download
```

### Verifying Installation

After installation, you can verify that everything is working correctly:

```bash
# Run the installation test
python test_installation.py
```

This script will check that all required dependencies can be imported and that the SpecViz package is properly installed.

### Makefile usage

[`Makefile`](https://github.com/specviz/specviz/blob/master/Makefile) contains a lot of functions for faster development.

<details>
<summary>1. Download and remove Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks coulb be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>3. Codestyle</summary>
<p>

Automatic formatting uses `pyupgrade`, `isort` and `black`.

```bash
make codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `isort`, `black` and `darglint` library

Update all dev libraries to the latest version using one comand

```bash
make update-dev-deps
```

<details>
<summary>4. Code security</summary>
<p>

```bash
make check-safety
```

This command launches `Poetry` integrity checks as well as identifies security issues with `Safety` and `Bandit`.

```bash
make check-safety
```

</p>
</details>

</p>
</details>

<details>
<summary>5. Type checks</summary>
<p>

Run `mypy` static type checker

```bash
make mypy
```

</p>
</details>

<details>
<summary>6. Tests with coverage badges</summary>
<p>

Run `pytest`

```bash
make test
```

</p>
</details>

<details>
<summary>7. All linters</summary>
<p>

Of course there is a command to ~~rule~~ run all linters in one:

```bash
make lint
```

the same as:

```bash
make test && make check-codestyle && make mypy && make check-safety
```

</p>
</details>

<details>
<summary>8. Docker</summary>
<p>

```bash
make docker-build
```

which is equivalent to:

```bash
make docker-build VERSION=latest
```

Remove docker image with

```bash
make docker-remove
```

More information [about docker](https://github.com/specviz/specviz/tree/master/docker).

</p>
</details>

<details>
<summary>9. Cleanup</summary>
<p>
Delete pycache files

```bash
make pycache-remove
```

Remove package build

```bash
make build-remove
```

Delete .DS_STORE files

```bash
make dsstore-remove
```

Remove .mypycache

```bash
make mypycache-remove
```

Or to remove all above run:

```bash
make cleanup
```

</p>
</details>

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/specviz/specviz/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

We use [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.

### List of labels and corresponding titles

|               **Label**               |  **Title in Releases**  |
| :-----------------------------------: | :---------------------: |
|       `enhancement`, `feature`        |       🚀 Features       |
| `bug`, `refactoring`, `bugfix`, `fix` | 🔧 Fixes & Refactoring  |
|       `build`, `ci`, `testing`        | 📦 Build System & CI/CD |
|              `breaking`               |   💥 Breaking Changes   |
|            `documentation`            |    📝 Documentation     |
|            `dependencies`             | ⬆️ Dependencies updates |

You can update it in [`release-drafter.yml`](https://github.com/specviz/specviz/blob/master/.github/release-drafter.yml).

GitHub creates the `bug`, `enhancement`, and `documentation` labels for you. Dependabot creates the `dependencies` label. Create the remaining labels on the Issues tab of your GitHub repository, when you need them.

## 🛡 License

[![License](https://img.shields.io/github/license/specviz/specviz)](https://github.com/specviz/specviz/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/specviz/specviz/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{specviz,
  author = {specviz},
  title = {viz of spectra},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/specviz/specviz}}
}
```

## Credits [![🚀 Your next Python package needs a bleeding-edge project structure.](https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen)](https://github.com/TezRomacH/python-package-template)

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template)

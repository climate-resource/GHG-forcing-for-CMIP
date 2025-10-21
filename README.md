<!--- --8<-- [start:description] -->
# GHG-forcing-for-CMIP

Produce CMIP7 GHG forcing data incl. earth observations.

**Key info :**
[![Docs](https://readthedocs.org/projects/ghg-forcing-for-cmip/badge/?version=latest)](https://ghg-forcing-for-cmip.readthedocs.io)
[![Main branch: supported Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fclimate-resource%2FGHG-forcing-for-CMIP%2Fmain%2Fpyproject.toml)](https://github.com/climate-resource/GHG-forcing-for-CMIP/blob/main/pyproject.toml)
[![Licence](https://img.shields.io/pypi/l/ghg-forcing-for-cmip?label=licence)](https://github.com/climate-resource/GHG-forcing-for-CMIP/blob/main/LICENCE)

**PyPI :**
[![PyPI](https://img.shields.io/pypi/v/ghg-forcing-for-cmip.svg)](https://pypi.org/project/ghg-forcing-for-cmip/)
[![PyPI install](https://github.com/climate-resource/GHG-forcing-for-CMIP/actions/workflows/install-pypi.yaml/badge.svg?branch=main)](https://github.com/climate-resource/GHG-forcing-for-CMIP/actions/workflows/install-pypi.yaml)

**Tests :**
[![CI](https://github.com/climate-resource/GHG-forcing-for-CMIP/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/climate-resource/GHG-forcing-for-CMIP/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/climate-resource/GHG-forcing-for-CMIP/branch/main/graph/badge.svg)](https://codecov.io/gh/climate-resource/GHG-forcing-for-CMIP)

**Other info :**
[![Last Commit](https://img.shields.io/github/last-commit/climate-resource/GHG-forcing-for-CMIP.svg)](https://github.com/climate-resource/GHG-forcing-for-CMIP/commits/main)
[![Contributors](https://img.shields.io/github/contributors/climate-resource/GHG-forcing-for-CMIP.svg)](https://github.com/climate-resource/GHG-forcing-for-CMIP/graphs/contributors)
## Status

<!---

We recommend having a status line in your repo
to tell anyone who stumbles on your repository where you're up to.
Some suggested options:

- prototype: the project is just starting up and the code is all prototype
- development: the project is actively being worked on
- finished: the project has achieved what it wanted
  and is no longer being worked on, we won't reply to any issues
- dormant: the project is no longer worked on
  but we might come back to it,
  if you have questions, feel free to raise an issue
- abandoned: this project is no longer worked on
  and we won't reply to any issues
-->

- prototype: the project is just starting up and the code is all prototype

<!--- --8<-- [end:description] -->

Full documentation can be found at:
[ghg-forcing-for-cmip.readthedocs.io](https://ghg-forcing-for-cmip.readthedocs.io/en/latest/).
We recommend reading the docs there because the internal documentation links
don't render correctly on GitHub's viewer.

## Installation

<!--- --8<-- [start:installation] -->
### As an application

If you want to use GHG-forcing-for-CMIP as an application,
then we recommend using the 'locked' version of the package.
This version pins the version of all dependencies too,
which reduces the chance of installation issues
because of breaking updates to dependencies.

The locked version of GHG-forcing-for-CMIP can be installed with

=== "pip"
    ```sh
    pip install 'ghg-forcing-for-cmip[locked]'
    ```

### As a library

If you want to use GHG-forcing-for-CMIP as a library,
for example you want to use it
as a dependency in another package/application that you're building,
then we recommend installing the package with the commands below.
This method provides the loosest pins possible of all dependencies.
This gives you, the package/application developer,
as much freedom as possible to set the versions of different packages.
However, the tradeoff with this freedom is that you may install
incompatible versions of GHG-forcing-for-CMIP's dependencies
(we cannot test all combinations of dependencies,
particularly ones which haven't been released yet!).
Hence, you may run into installation issues.
If you believe these are because of a problem in GHG-forcing-for-CMIP,
please [raise an issue](https://github.com/climate-resource/GHG-forcing-for-CMIP/issues).

The (non-locked) version of GHG-forcing-for-CMIP can be installed with

=== "pip"
    ```sh
    pip install ghg-forcing-for-cmip
    ```

Additional dependencies can be installed using

=== "pip"
    ```sh
    # To add plotting dependencies
    pip install 'ghg-forcing-for-cmip[plots]'

    # To add all optional dependencies
    pip install 'ghg-forcing-for-cmip[full]'
    ```

### For developers

For development, we rely on [uv](https://docs.astral.sh/uv/)
for all our dependency management.
To get started, you will need to make sure that uv is installed
([instructions here](https://docs.astral.sh/uv/getting-started/installation/)
(we found that the self-managed install was best,
particularly for upgrading uv later).

For all of our work, we use our `Makefile`.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone.
In order to create your environment, run `make virtual-environment`.

If there are any issues, the messages from the `Makefile` should guide you through.
If not, please raise an issue in the
[issue tracker](https://github.com/climate-resource/GHG-forcing-for-CMIP/issues).

For the rest of our developer docs, please see [development][development].

<!--- --8<-- [end:installation] -->

## Producing the forcings

In order to produce the forcings, you need to run the notebooks.
The easiest way to do this is with `make docs`.
For this to work, you will also need to have a local prefect server running.
Start this in a separate terminal with, `uv run prefect server start`.

### Prefect set up

[Maybe delete this, but I got myself in a big mess with this]
If you use prefect in more than one project,
you can get yourself in a mess.
(This is basically because prefect assumes it will run on a server I think
i.e. one config per machine.
By using it for multiple projects on the same machine,
we're breaking this a bit.)

To avoid clashes, you will likely want to make a
[profile](https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings)
specific to this project, e.g.

```
uv run prefect profile create ghg-forcing-for-cmip
```

Then use it with

```
uv run prefect profile use ghg-forcing-for-cmip
```

If you want the API to run in 'ephemeral' mode
i.e. start up each time, add

```
uv run prefect config set PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=True
```

To avoid clashes with other databases,
tell prefect to use a database specific to this project

```
mkdir .prefect
uv run prefect config set PREFECT_API_DATABASE_CONNECTION_URL='sqlite+aiosqlite:////path/to/this/repo/.prefect/prefect.db'
```

### Registering with ECMWF data stores

In order for this to work, you need to follow
[the ECMWF set up instructions](https://cds.climate.copernicus.eu/how-to-api).
The instructions are not super clear,
so here are some clarifications.

The credentials go in `$HOME/.ecmwfdatastoresrc`.
On unix-systems, this is `~/.ecmwfdatastoresrc`.

In this file, you should have the following content

```
url: https://cds.climate.copernicus.eu/api
key: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

The key is the ECMWF datastores key.
This is the same content as the content you get if you follow
the instructions at
https://cds.climate.copernicus.eu/how-to-api,
the file it needs to go into is just named differently.

[@Flo as a note, see how the ECMWF datastore is also a rest API.
They probably have POST/PATCH/DELETE end points too
so they can manage adding and updating existing data,
but obviously us users can't access them,
we can probably only hit the GET end points :)]

The first time you run, you will hit errors of the form

```
HTTPError: 403 Client Error: Forbidden for url: https://cds.climate.copernicus.eu/api/retrieve/v1/processes/satellite-methane/execution
required licences not accepted
Not all the required licences have been accepted; please visit https://cds.climate.copernicus.eu/datasets/satellite-methane?tab=download#manage-licences to accept the required licence(s).
```

Follow the links provided and accept the licences, then run again.
(Yes, this is a bit stupid: an API that still requires manual acceptance.
That's out of our control and fortunately we only have to do this once per token,
so isn't a blocker for e.g. running in CI.)

[For some reason, when I re-run the notebook, I don't get any prefect caching.
Not sure if it's my fault because of the prefect set up stuff above
or an issue with how this has been set up.
TODO: make an issue then decide whether we tackle this now or later.
Decision question: does Flo have/care about this issue or not?]

## Original template

This project was generated from this template:
[copier core python repository](https://gitlab.com/openscm/copier-core-python-repository).
[copier](https://copier.readthedocs.io/en/stable/) is used to manage and
distribute this template.

from pathlib import Path
from typing import NamedTuple

import setuptools
import toml

ALL_SECTION = "all-plugins"
DEV_SECTION = "dev"
CORE_PACKAGE_NAME = "agents-core"
PLUGINS_DIR = "plugins"

PLUGINS_TO_IGNORE = []


def main():
    """
    Validate that all namespace packages are include into optional dependencies in "agents-core/pyproject.toml".

    The script must be executed from the project root.
    """

    # First, validate that the script is executed from the project's root
    if not _cwd_is_root():
        raise RuntimeError("The script must be executed from the project root.")

    # Get all namespace packages in plugins/
    plugins = setuptools.find_namespace_packages(PLUGINS_DIR)
    plugins_roots = {p.split(".")[0] for p in plugins}
    plugins_packages = [_get_plugin_package_name(plugin) for plugin in plugins_roots]

    # Exclude some plugins from analysis if defined
    plugins_packages = [p for p in plugins_packages if p not in PLUGINS_TO_IGNORE]

    # Get optional dependencies for "agents-core" package.
    core_optional_dependencies = _get_core_optional_dependencies()

    # Validate that "agents-core" has "all-plugins" section in optional dependencies
    if not core_optional_dependencies.all:
        raise PyprojectValidationError(
            f'Optional dependencies for "{CORE_PACKAGE_NAME}" are missing the "{ALL_SECTION}" section.'
        )

    # Validate that all available plugins are listed in "all-plugins"
    not_included_in_all = set(plugins_packages) - set(core_optional_dependencies.all)
    if not_included_in_all:
        raise PyprojectValidationError(
            f'The following plugins are not included in the "{ALL_SECTION}" '
            f'section in "{CORE_PACKAGE_NAME}" package: {", ".join(not_included_in_all)}"'
        )

    # Validate that every plugin has a dedicated section in core's optional dependencies
    plugins_sections_reversed = {
        tuple(v): k for k, v in core_optional_dependencies.plugins.items()
    }
    plugins_without_optional = []
    for package_name in plugins_packages:
        if (package_name,) not in plugins_sections_reversed:
            plugins_without_optional.append(package_name)

    if plugins_without_optional:
        raise PyprojectValidationError(
            f"The following plugins do not have an optional dependency section "
            f'in "{CORE_PACKAGE_NAME}" package: \n{", ".join(plugins_without_optional)}". \n\n'
            f'To fix it, add a section for each plugin to [project.optional-dependencies] inside "{CORE_PACKAGE_NAME}/pyproject.toml" like this: \n\n'
            f'plugin_name = ["vision-agents-plugins-plugin-name"]'
        )
    return None


class PyprojectValidationError(Exception): ...


class CoreDependencies(NamedTuple):
    all: list[str]
    plugins: dict[str, list[str]]


def _cwd_is_root():
    cwd = Path.cwd()
    return (cwd / CORE_PACKAGE_NAME).exists() and (cwd / PLUGINS_DIR).exists()


def _get_plugin_package_name(plugin: str) -> str:
    with open(PLUGINS_DIR / Path(plugin) / "pyproject.toml", "r") as f:
        pyproject = toml.load(f)
    return pyproject["project"]["name"]


def _get_core_optional_dependencies() -> CoreDependencies:
    with open(Path(CORE_PACKAGE_NAME) / "pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    optionals: dict[str, list[str]] = pyproject.get("project", {}).get(
        "optional-dependencies", {}
    )
    if not optionals:
        raise PyprojectValidationError(
            f'No optional dependencies found for "{CORE_PACKAGE_NAME}".'
        )
    optionals_all = optionals.get(ALL_SECTION, [])
    optionals_plugins = {
        k: v for k, v in optionals.items() if k not in (ALL_SECTION, DEV_SECTION)
    }
    return CoreDependencies(all=optionals_all, plugins=optionals_plugins)


if __name__ == "__main__":
    main()

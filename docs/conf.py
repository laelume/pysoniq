# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# -- Path setup --------------------------------------------------------------
# Adjust if your package lives under src/ — relative only, no hardcoded paths.
sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(0, os.path.abspath("../src"))  # FALLBACK: src-layout

# -- Project information ------------------------------------------------------
project = "pysoniq"
author = "laelume"
copyright = "2025-2026, laelume"
license = "MIT"

# -- Version: prefer live scm resolution, fall back to installed metadata ----
try:
    from setuptools_scm import get_version
    release = get_version(root="..", relative_to=__file__)
except Exception:  # FALLBACK: scm unavailable (e.g. sdist build, no .git)
    from importlib.metadata import version as _v, PackageNotFoundError
    try:
        release = _v("pysoniq")
    except PackageNotFoundError:
        release = "0.0.0"

version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

autosummary_generate = True

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Heavy/optional imports so autodoc doesn't need them at build time ---
autodoc_imports = [
    "numpy>=1.19.0",
    "scipy",
    "joblib"
]

# -- HTML output --------------------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]


# run apidoc at the start of every build via hook
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    import os
    here = os.path.abspath(os.path.dirname(__file__))
    packages = {
        "pysoniq": "../pysoniq",
    }
    for name, path in packages.items():
        out = os.path.join(here, "api", name)
        src = os.path.abspath(os.path.join(here, path))
        main(["-o", out, src, "--separate", "--module-first", "--force"])

def setup(app):
    app.connect("builder-inited", run_apidoc)
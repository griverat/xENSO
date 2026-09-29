"""Sphinx configuration for the xENSO documentation, modelled on xarray's."""

import datetime
import shutil
from pathlib import Path

import xenso

# -- Project information -----------------------------------------------------

project = "xENSO"
author = "Gerardo A. Rivera"
copyright = f"2021-{datetime.date.today().year}, {author}"
version = release = xenso.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["how2package4ioos.md", "_build"]
language = "en"

extlinks = {
    "issue": ("https://github.com/DangoMelon/xENSO/issues/%s", "GH%s"),
    "pull": ("https://github.com/DangoMelon/xENSO/pull/%s", "PR%s"),
}

# -- Autodoc and autosummary -------------------------------------------------

autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

# -- Notebooks ---------------------------------------------------------------

# The tutorial lives in notebooks/; copy it here so it can be rendered, without executing it
shutil.copy(
    Path(__file__).parent / "../../notebooks/tutorial.ipynb", Path(__file__).parent / "tutorial.ipynb"
)
nb_execution_mode = "off"

# -- Copy button -------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.{3,}: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "xENSO"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/DangoMelon/xENSO",
    "use_edit_page_button": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "footer_center": ["last-updated"],
}
html_context = {
    "github_user": "DangoMelon",
    "github_repo": "xENSO",
    "github_version": "master",
    "doc_path": "docs/source",
}
html_last_updated_fmt = "%Y-%m-%d"

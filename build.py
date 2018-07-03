from pybuilder.core import use_plugin, init

use_plugin("python.core")


# this plugin allows installing project dependencies with pip
use_plugin("python.install_dependencies")

# the python unittest plugin allows running python's standard library unittests
# use_plugin("python.unittest")

# a linter plugin that runs flake8 (pyflakes + pep8) on our project sources
use_plugin("python.flake8")

# a plugin that measures unit test statement coverage
use_plugin("python.coverage")

# The project name
name = "causality"

default_task = ['install_dependencies', 'publish']


@init
def set_properties(project):
    project.set_property("coverage_exceptions", ['__init__', 'main'])
    project.set_property("dir_dist_scripts", [])
    project.depends_on_requirements("requirements.txt")

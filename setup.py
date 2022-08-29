import os
from setuptools import setup, find_packages


__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "diffstar", "_version.py"
)
with open(pth, "r") as fp:
    exec(fp.read())


setup(
    name="diffstar",
    version=__version__,
    author=["Alex Alarcon", "Andrew Hearin"],
    author_email=["alexalarcongonzalez@gmail.com", "ahearin@anl.gov"],
    description="Differentiable star formation histories",
    install_requires=["numpy", "jax", "diffmah"],
    packages=find_packages(),
    package_data={"diffstar": ["tests/testing_data/*.txt"]},
    url="https://github.com/ArgonneCPAC/diffstar",
)

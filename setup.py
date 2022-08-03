from setuptools import setup, find_packages


PACKAGENAME = "diffstar"
VERSION = "0.0.2"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=["Alex Alarcon", "Andrew Hearin"],
    author_email=["alexalarcongonzalez@gmail.com", "ahearin@anl.gov"],
    description="Differentiable star formation histories",
    long_description="Differentiable star formation histories",
    install_requires=["numpy", "jax", "diffmah"],
    packages=find_packages(),
    package_data={"diffstar": ["tests/testing_data/*.txt"]},
    url="https://github.com/ArgonneCPAC/diffstar",
)

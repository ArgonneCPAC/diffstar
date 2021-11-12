from setuptools import setup, find_packages


PACKAGENAME = "diffstar"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=["Alex Alarcon", "Andrew Hearin"],
    author_email=["alexalarcongonzalez@gmail.com", "ahearin@anl.gov"],
    description="Differentiable star formation histories",
    long_description="Differentiable star formation histories",
    install_requires=["numpy", "jax"],
    packages=find_packages(),
    url="https://github.com/ArgonneCPAC/diffstar",
)

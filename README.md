# diffstar

## Installation
For a typical development environment in conda:

```
$ conda config --add channels conda-forge
$ conda config --prepend channels conda-forge
$ conda create -n diffit python=3.9 numpy numba flake8 pytest jax ipython jupyter matplotlib scipy h5py diffmah
```

To install diffstar into your environment from the source code:
```
$ conda activate my_env_name
$ cd /path/to/root/diffstar
$ python setup.py install
```

Data for this project can be found [at this URL](https://portal.nersc.gov/project/hacc/aphearin/diffstar_data/).

## Scripts and demo notebooks
The `diffstar_fitter_demo.ipynb` notebook demonstrates how to fit the SFH of a simulated galaxy with a diffstar approximation.

See `history_fitting_script.py` for an example of how to fit the SFHs of a large number of simulated galaxies in parallel with mpi4py.

## Citing diffstar
You can find a preprint of the diffstar paper in [arXiv](https://arxiv.org/abs/2205.04273). Citation information for the paper can be found at [this ADS link](https://ui.adsabs.harvard.edu/abs/2022arXiv220504273A/abstract), copied below for convenience:

```
@ARTICLE{2022arXiv220504273A,
       author = {{Alarcon}, Alex and {Hearin}, Andrew P. and {Becker}, Matthew R. and {Chaves-Montero}, Jon{\'a}s},
        title = "{Diffstar: A Fully Parametric Physical Model for Galaxy Assembly History}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2022,
        month = may,
          eid = {arXiv:2205.04273},
        pages = {arXiv:2205.04273},
archivePrefix = {arXiv},
       eprint = {2205.04273},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220504273A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```

# diffstar

Code documentation can be found at [diffstar.readthedocs.io](https://diffstar.readthedocs.io/en/latest/).

## Installation
The latest release of diffstar is available for installation with either pip or conda-forge:
```
$ conda install -c conda-forge diffstar
```

Alternatively, to install diffstar into your environment from the source code:

```
$ cd /path/to/root/diffstar
$ pip install .
```

### Environment configuration
For a typical development environment with conda-forge:

```
$ conda create -c conda-forge -n diffit python=3.11 numpy numba flake8 pytest jax ipython jupyter matplotlib scipy h5py diffmah diffstar
```

## Project data
Data for this project can be found [at this URL](https://portal.nersc.gov/project/hacc/aphearin/diffstar_data/).

## Scripts and demo notebooks

The `demo_diffstar_sfh.ipynb` notebook in the `docs` folder illustrates how to use the Diffstar model for individual SFH, and how to generate the SFH of a population of galaxies using DiffstarPop.

See `diffstar_fitting_script_umachine_mgash.py` for an example of how to fit the SFHs of a large number of simulated galaxies in parallel with mpi4py.

The `diffstar_fitter_demo.ipynb` notebook demonstrates how to fit the SFH of a simulated galaxy with a diffstar approximation.


See `fit_mstar_ssfr_pdfs_mgash.py` for an example of how to use DiffstarPop to fit a set of Mstar and sSFR PDFs, and `measure_smhm_smdpl_script_mpi_mgash.py` for an example of how to generate the target data from a set of Diffstar fits.

## Citing diffstar
[The Diffstar paper](https://arxiv.org/abs/2205.04273) has been published in [Monthly Notices of the Royal Astronomical Society](https://academic.oup.com/mnras/article-abstract/518/1/562/6795944?redirectedFrom=fulltext). Citation information for the paper can be found at [this ADS link](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518..562A/abstract), copied below for convenience:

```
@ARTICLE{2023MNRAS.518..562A,
       author = {{Alarcon}, Alex and {Hearin}, Andrew P. and {Becker}, Matthew R. and {Chaves-Montero}, Jon{\'a}s},
        title = "{Diffstar: a fully parametric physical model for galaxy assembly history}",
      journal = {MNRAS},
     keywords = {galaxies: evolution, galaxies: fundamental parameters, galaxies: star formation, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2023,
        month = jan,
       volume = {518},
       number = {1},
        pages = {562-584},
          doi = {10.1093/mnras/stac3118},
archivePrefix = {arXiv},
       eprint = {2205.04273},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.518..562A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

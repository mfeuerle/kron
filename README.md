# kron

This is an implicit implementation of Kronecker products, i.e. the products are never assembled, and its parts are stored separately. This is much more memory efficient and allows for a faster matrix-vector multiplication. The documentation can be found [here](https://mfeuerle.github.io/kron/). Was tested using `python=3.12.13`, `numpy=1.26.4`, `scipy=1.16.3`.

## Installing

To install this project:

```
conda env create -f environment.yml
conda activate kron
```

## Building the Docs

```
cd docs
make html
```

# kron

This is an implicit implementation of Kronecker products, i.e. the products are never assembled, and its parts are stored separately. This is much more memory efficient and allows for a faster matrix-vector multiplication. The documentation can be found [here](https://mfeuerle.github.io/kron/).

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

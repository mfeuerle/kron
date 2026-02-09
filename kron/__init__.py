# Moritz Feuerle, May 2022

r""" 
===============================
:mod:`kron`
===============================


Matrix classes
--------------
.. autosummary::
   :toctree: generated/
   
   kron_base
   kron_matrix
   kronsum_matrix
   kronblock_matrix
   block_matrix
   
   
Construction functions
----------------------
.. autosummary::
   :toctree: generated/
   
   kron
   block
   hstack
   vstack
   blockedkron
   diag
   eye
   zeros
   ones
   
   
Type identification functions
-----------------------------
.. autosummary::
   :toctree: generated/
   
   iskronrelated
   iskronshaped
   iskron
   iskronsum
   isblock
   iskronblock
   
   
      
   
Inheritance diagram
-------------------
   
.. inheritance-diagram:: kron._kron
   :private-bases:
   :parts: 2



.. todo:: 
   type hints for all methods
.. todo::
   switch to @property syntax for all attributes
.. todo::
   restructure with block_matrix, sum_matrix, kron_matrix, which can be used to build combinations like kronsum, blocksum, kronblock, kronblocksum
   
   
.. sectionauthor:: Moritz Feuerle, 2022
"""
from ._kron import *
from ._construct import *
from ._interfaces import kron_base, iskronrelated, iskronshaped






replaced_operators = (
    "__add__", "__sub__",
    "__eq__", "__ne__", "__ge__", "__gt__", "__le__", "__lt__", 
    "__matmul__", 
    "__mul__", "__div__", "__truediv__",
)

from ._dirty_patches  import _patch_priority
import scipy.sparse as _sparse
from scipy.sparse._base import _spbase

sparse_types = [_spbase] + [type_ for type_ in _sparse.__dict__.values() if isinstance(type_, type) and issubclass(type_, _spbase)]
_patch_priority(sparse_types, "__sparse_priority__", replaced_operators)

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



def _patch_scipy_sparse_priority():
    """Patch scipy.sparse to respect ``__sparse_priority__``.
    
    This is necessary to use ``__r*__`` methods like ``__rmatmul__`` from custom classes
    instead of ``__*__`` methods like ``__matmul__`` from scipy sparse matrices,
    since scipy sparse matrices will never return :const:`NotImplemented`.
    
    This function replaces operators like ``__matmul__`` or ``__eq__`` of
    scipy sparse base classes and all their subclasses. In the new operator it is first 
    checked if ``other`` has the attribute ``__sparse_priority__`` with a higher value than 
    the sparse matrix's ``__sparse_priority__``. If so, :const:`NotImplemented`
    is returned. Otherwise the original implementation of the operator is called.

    This workaround is based on:
    https://github.com/scipy/scipy/issues/4819#issuecomment-920722279
    """
    from scipy.sparse._base import _spbase
    
    replaced_operators = (
        "__add__", "__sub__",
        "__eq__", "__ne__", "__ge__", "__gt__", "__le__", "__lt__", 
        "__matmul__", 
        "__mul__", "__div__", "__truediv__",
    )

    def teach_sparse_priority(operator):
        def respect_sparse_priority(self, other):
            self_priority = getattr(self, "__sparse_priority__", 0.0)
            other_priority = getattr(other, "__sparse_priority__", -1.0)
            if self_priority < other_priority:
                return NotImplemented
            else:
                return operator(self, other)
        return respect_sparse_priority

    # Patch the internal _spbase class which is the actual base for all sparse types
    # This is where __matmul__ and _matmul_dispatch are defined
    if not hasattr(_spbase, '__sparse_priority__'):
        _spbase.__sparse_priority__ = 0.0
    
    for operator_name in replaced_operators:
        if hasattr(_spbase, operator_name):
            operator = getattr(_spbase, operator_name)
            if not getattr(operator, '_sparse_priority_wrapped', False):
                wrapped_operator = teach_sparse_priority(operator)
                wrapped_operator._sparse_priority_wrapped = True
                setattr(_spbase, operator_name, wrapped_operator)


# Apply the patch immediately on import
_patch_scipy_sparse_priority()

from ._kron import *
from ._construct import *
from ._interfaces import kron_base, iskronrelated, iskronshaped
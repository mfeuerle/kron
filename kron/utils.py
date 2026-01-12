# Moritz Feuerle, May 2022

r"""
=====================================
:mod:`kron.utils`
=====================================

Functions
--------------------

.. autosummary::
    :toctree: generated/
   
    tosparray
   
Matrix collection
--------------------

.. autosummary::
    :toctree: generated/
    
    MatrixCollection
    iscollection
    

.. sectionauthor:: Moritz Feuerle, Apr 2022
"""

import scipy.sparse
import numpy as np
from . import _interfaces


__all__ = ['tosparray', 'MatrixCollection', 'iscollection']


def tosparray(x):
    """Simple wrapper to wrap deprecated :obj:`scipy.sparse.spmatrix` like :obj:`scipy.sparse.coo_matrix`
    in :obj:`scipy.sparse.sparray` like :obj:`scipy.sparse.coo_array`.
    
    Parameters
    ----------
    x
        matrix to be changed into :obj:`scipy.sparse.sparray`
        
    Returns
    -------
    :obj:`scipy.sparse.sparray`
        :obj:`scipy.sparse.sparray` representation of ``x`` if possible, or ``x``
        
    Examples
    --------
    >>> from scipy.sparse import coo_matrix
    >>> from kron._kron import tosparray
    >>> type(tosparray(coo_matrix([[5]])))
    <class 'scipy.sparse._arrays.coo_array'>
    
    >>> from scipy.sparse import coo_array
    >>> from kron._kron import tosparray
    >>> type(tosparray(coo_array([[5]])))
    <class 'scipy.sparse._arrays.coo_array'>
    
    >>> import numpy as np
    >>> from kron._kron import tosparray
    >>> type(tosparray(np.array([[5]])))
    <class 'numpy.ndarray'>
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    if not scipy.sparse.issparse(x):
        return x
    cls = getattr(scipy.sparse, f'{x.format}_array')
    return cls(x) 



class MatrixCollection:
    """Just a small wraper to a list of matricies with common shape.
    
    Can be used as 
    
    >>> M = MatrixCollection(A,B,C)
    >>> for A in M:
    >>>     print(A)
    >>> for i in range(len(M)):
    >>>     print(M[i])
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    
    __kron_priority__ = 0.1
    __array_priority__ = 100
    __sparse_priority__ = 100
    
    dtype : np.dtype
    """Common datatype of a single matrix element over all matricies"""
    
    shape : tuple
    """Shape of the matricies"""
    
    ndim : int = 2
    """Number of dimensions of the matricies"""
    
    data : list
    """Matrix collection"""
    
    def __init__(self, *A):
        """
        Parameters
        ----------
        A 
            Collection of matricies
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if len(A) == 1:
            A = A[0]
            
        shape = A[0].shape
        ndim  = A[0].ndim
        
        assert all(a.shape == shape for a in A)
        assert all(a.ndim == ndim for a in A)
        
        self.dtype = np.result_type(a.dtype for a in A)
        self.shape = shape
        self.ndim = ndim
        self.data = [a for a in A]

        
    def __len__(self):
        """Number of collected matricies.

        Returns
        -------
        int
            length of the collection
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """Returnes the given elements of the collection

        Parameters
        ----------
        idx
            Index of the matricies that will be returned

        Returns
        -------
        :obj:`MatrixCollection` or single matrix
            returns a single matrix if ``idx`` is scalar, otherwise a new collection
            
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        out = self.data[idx]
        return MatrixCollection(out) if isinstance(out, list) else out
    
    def __setitem__(self, index, value):
        """Inplace operation, replacing some of the collected matricies.

        Parameters
        ----------
        index
            matricies to be replaced
        value
            new matricies
            
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        self.data[index] = value
        assert all(A.shape == self.shape for A in self.data)
        assert all(A.ndim == self.ndim for A in self.data)
        self.dtype = np.result_type(a.dtype for a in self.data)
    
    def __iter__(self):
        """Iterates over all collected matricies.

        Yields
        ------
        matrix
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        for A in self.data:
            yield A
            
            
    def append(self, other):
        """Expands the matrix collection.

        Parameters
        ----------
        other : list or :obj:`MatrixCollection`
            collection of matricies that will be added

        Returns
        -------
        :obj:`MatrixCollection`
            New expanded collection
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if iscollection(other):
            return MatrixCollection(self.data + other.data)
        elif hasattr(other,'ndim') and hasattr(other,'shape') and other.ndim==self.ndim and other.shape==self.shape:
            return MatrixCollection(self.data + [other])
        else:
            return MatrixCollection(self.data + other)
        
            
    def __add__(self, other):
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        if other == 0:
            return MatrixCollection(self.data)
        return self.append(other)
    __add__.__doc__ = append.__doc__

    
    def __radd__(self, other):
        if other == 0:
            return MatrixCollection(self.data)
        return self.append(other)
    __radd__.__doc__ = append.__doc__
    

    

            
    def __matmul__(self, other):
        """Matrix-matrix or matrix vector multiplication, applied to all matricies in the collection.

        Parameters
        ----------
        other
            matrix or vector; will be multiplied from the rigth to all matricies

        Returns
        -------
        :obj:`MatrixCollection`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        return MatrixCollection([A @ other for A in self.data])
    
    def __rmatmul__(self, other):
        """Matrix-matrix or matrix vector multiplication, applied to all matricies in the collection.

        Parameters
        ----------
        other
            matrix or vector; will be multiplied from the left to all matricies

        Returns
        -------
        :obj:`MatrixCollection`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return MatrixCollection([other @ A for A in self.data])
    
    def __mul__(self, other):
        """Elementwise multiplication, applied to all matricies in the collection.

        Parameters
        ----------
        other
            matrix, vector or scalar; will be multiplied from the left to all matricies

        Returns
        -------
        :obj:`MatrixCollection`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        return MatrixCollection([A * other for A in self.data])
    
    def __rmul__(self, other):
        """Elementwise multiplication, applied to all matricies in the collection.

        Parameters
        ----------
        other
            matrix, vector or scalar; will be multiplied from the right to all matricies

        Returns
        -------
        :obj:`MatrixCollection`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return MatrixCollection([other * A for A in self.data])
            
    def __repr__(self):
        M,N = self.shape
        return '<%dx%d %s of %s matricies>' % (M, N, self.__class__.__name__, len(self))
    
    
def iscollection(x):
    """Is ``x`` a :obj:`MatrixCollection`?
    
    Parameters
    ----------
    x
        object to check for being a collection of matricies
        
    Returns
    -------
    bool
        ``True`` if ``x`` is a matrix collection, ``False`` otherwise
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return isinstance(x, MatrixCollection)
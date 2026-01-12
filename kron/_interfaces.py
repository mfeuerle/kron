# Moritz Feuerle, May 2022

import numpy as np
from scipy.sparse import issparse

__all__ = ['kron_base', 'iskronrelated', 'iskronshaped']


class kron_base:
    """Base class for Kronecker product related matricies. 
    The main purpose of this interface is to identify matricies from this framwork
    in a unified manner and to define some alias methods.
    
    To implement this initerface the following methods need to be provided:
        - :meth:`__mul__`, :meth:`__matmul__`, :meth:`__add__` and :meth:`transpose` for matrix operations.
        (A default implementation of :meth:`__rmatmul__` is provided but should be 
        implemented as well if possible.)
        - :meth:`getklayer` for accessing all matricies in each Kronecker layer.
        - :meth:`asformat` for exporting the matrix in commen formats like :obj:`numpy.ndarray` or
        :obj:`scipy.sparse.sparray`.
    """
    
    dtype : np.dtype
    """Datatype of a single matrix element"""
    
    shape : tuple
    """Shape of the matrix"""
    
    ndim : int = 2
    """Number of dimensions (this is always 2)"""
    
    kdim : int
    r"""Number of concatinated Kronecker products (this is always 1).
    
    Example: :math:`A \otimes B \otimes C` has ``kdim=3``."""
    
    __array_priority__ = 11  # enables __rmatmul__ with ndarray
    __sparse_priority__ = 11  # enables __rmatmul__ with scipy sparse matrices
    
    def __init__(self):
        """This is a interface, do not instantiate this class directly.

        Raises:
            ValueError: allways returned if this class is instantiate directly.
        """
        raise ValueError("This class is a interface and is not to be instantiated directly.")
    
    
    ####################################################
    # element-wise multiplication
    ####################################################
    def __mul__(self, other):       # self * other
        """Elementwise multiplication ``self * other`` of two matricies ``self``
        and ``other`` with same shape, or scalar multiplication where ``other`` is a scalar value.
        
        Warning
        ----------
        This is a interface method, and has to be implemented.
        """
        return NotImplementedError(" This is a interface method and has to be implemented.")
    
    def __rmul__(self, other):      # other * self
        """Same as :meth:`self * other <__mul__>` since elementwise multiplication is commutative.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__mul__(other)
    
    def multiply(self, other):      # self * other
        """Elementwise multiplication :meth:`self * other <__mul__>`.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__mul__(other)
    
    ####################################################
    # Matrix-Matrix and Matrix-Vector multiplication
    ####################################################
    def __matmul__(self, other):    # self @ other
        """Matrix-matrix or matrix-vector multiplication ``self @ other`` of the matrix 
        ``self`` with another matirx or vector ``other``.
        
        Warning
        ----------
        This is a interface method, and has to be implemented.
        """
        return NotImplementedError(" This is a interface method and has to be implemented.")
    
    def __rmatmul__(self, other):   # other @ self
        """Left matrix-matrix or vector-matrix multiplication ``other @ self`` of the matrix 
        ``self`` with another matirx or vector ``other``.
        For more informations see :meth:`self @ other <__matmul__>`.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return ( self.transpose() @ other.transpose() ).transpose()
    
    def dot(self, other):
        """Dot-product :meth:`self @ other <__matmul__>`.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__matmul__(other)
    
    def rdot(self, other):
        """Right dot-product :meth:`other @ self <__matmul__>`.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__rmatmul__(other)
    
    ####################################################
    # Addition
    ####################################################
    def __add__(self, x):
        """Addition ``self + other``.
        
        Warning
        ----------
        This is a interface method, and has to be implemented.
        """
        return NotImplementedError(" This is a interface method and has to be implemented.")
    
    def __radd__(self, other):  # other + self
        """Addition ``other + self``; see :meth:`__add__`.
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__add__(other)
    
    def __sub__(self,other):    # self - other
        """Subtraction ``self - other``; see :meth:`__add__`.
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if hasattr(other, "__array_priority__") and self.__array_priority__ < other.__array_priority__:
            return NotImplemented
        return self.__add__(-other)  
    
    def __rsub__(self,other):   # other - self
        """Subtraction ``other - self``; see :meth:`__add__`.
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return (-self).__radd__(other)   
    
    def __neg__(self):          # -self
        """Negation ``-self``.
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__mul__(-1)
    
    def sum(self, axis=None):
        """Sum of matrix elements over a given axis.

        Parameters
        ----------
        axis : None or int, optional
            Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return NotImplementedError(" This is a interface method and has to be implemented.")

    ####################################################
    # Adjoint matrix
    ####################################################
    def transpose(self):
        """Returnes the transposed of the matrix.
        
        Warning
        ----------
        This is a interface method, and has to be implemented.
        """
        pass
    
    @ property
    def T(self):
        """ Matrix transposed, shortcut for :meth:`transpose`.
        """
        return self.transpose()


    ####################################################
    # access kronecker layers
    ####################################################
    def getklayer(self, k):
        """Returnes all matricies that build the ``k``-th Kronecker layer.
        
        Parameters
        ----------
        k
            Kronecker layer (``0 <= k <`` :attr:`kshape`)
        
        Warning
        ----------
            This is a interface method, and has to be implemented.
        """
        return NotImplementedError(" This is a interface method and has to be implemented.")
        
    ####################################################
    # formation export
    ####################################################
    def asformat(self, format, copy=False):
        r"""Assambles the full Kronecker product matrix in the given format.

        Parameters
        ----------
        format
            Either ``'array'`` or ``'dense'`` to return a dense :obj:`numpy.ndarray`,
            one of the sparse matrix formats from :mod:`scipy.sparse`
            
            * ``'csc'``: Compressed Sparse Column
            * ``'csr'``: Compressed Sparse Row
            * ``'bsr'``: Block Sparse Row
            * ``'coo'``: COOrdinate
            * ``'dia'``: DIAgonal
            * ``'dok'``: Dictionary Of Keys
            * ``'lil'``: List of Lists
            
            to return a :obj:`scipy.sparse.sparray` or ``None`` to assemble the matrix in any format.
            Other sparse formats from :const:`scipy.sparse._base._formats` are in general not supported.
        copy
            Just for consistency with :meth:`scipy.sparse.sparray.asformat`. This method will alwas copy.
            
        Warning
        ----------
        Assembeling the full Kronecker product matrix massively increases memory consumption.
            
        .. todo:: 
            für None: Dens umschreiben in sparse falls dünn besetz und umgekehrt.

        Returns
        ----------
        :obj:`numpy.ndarray` or :obj:`scipy.sparse.sparray`
            Full Kronecker product matrix in the given format.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        pass
    
    def assemble(self, ifsparse=None):
        """Assembles the Kronecker product matrix in any format; 
        same as :meth:`asformat(None)<asformat>`. Additionally a sparse 
        format could be supplied which will be used if a sparse matrix would be returned.
        
        Parameters
        ----------
        ifsparse
            If the returned matrix is sparse, this format will be used.
        
        Returns
        ----------
        :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        A = self.asformat(None)
        if issparse(A) and ifsparse is not None:
            A = A.asformat(ifsparse)
        return A
    
    def toarray(self):
        """Assembles the Kronecker product matrix as dense array; 
        same as :meth:`asformat('array')<asformat>`.
        
        Returns
        ----------
        :obj:`numpy.ndarray`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('array')
    
    def todense(self):
        """Assembles the Kronecker product matrix as dense array; 
        same as :meth:`asformat('dense')<asformat>`.
        
        Returns
        ----------
        :obj:`numpy.ndarray`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('dense')
    
    def tobsr(self):
        """Assembles the Kronecker product matrix in Block Sparse Row format; 
        same as :meth:`asformat('bsr')<asformat>`.
        
        Returns
        ----------
        :obj:`scipy.sparse.bsr_array`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('bsr')
    
    def tocoo(self):
        """Assembles the Kronecker product matrix in COOrdinate format; 
        same as :meth:`asformat('coo')<asformat>`.
        
        Returns
        ----------
        :obj:`scipy.sparse.coo_array`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('coo')
    
    def tocsc(self):
        """Assembles the Kronecker product matrix in Compressed Sparse Column format; 
        same as :meth:`asformat('csc')<asformat>`.
        
        Returns
        ----------
        :obj:`scipy.sparse.csc_array`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('csc')
    
    def tocsr(self):
        """Assembles the Kronecker product matrix in Compressed Sparse Row format; 
        same as :meth:`asformat('csr')<asformat>`.
        
        Returns
        ----------
        :obj:`scipy.sparse.csr_array`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('csr')
    
    def todia(self):
        """Assembles the Kronecker product matrix in DIAgonal format; 
        same as :meth:`asformat('dia')<asformat>`.
        
        Returns
        ----------
        :obj:`scipy.sparse.dia_array`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('dia')
    
    def todok(self):
        """Assembles the Kronecker product matrix in Dictionary Of Keys format; 
        same as :meth:`asformat('dok')<asformat>`.
        
        Returns
        ----------
        :obj:`scipy.sparse.dok_array`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('dok')
    
    def tolil(self):
        """Assembles the Kronecker product matrix in List of Lists format; 
        same as :meth:`asformat('lil')<asformat>`.
        
        Returns
        ----------
        :obj:`scipy.sparse.lil_array`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.asformat('lil')   
    
    ####################################################
    # Interface for aslinearoperator
    ####################################################
    def matvec(self, x):
        """Matrix-vector multiplication ``A x``, implements :meth:`scipy.sparse.linalg.LinearOperator.matvec`.
        
        Note
        ----------
        This method is only here for :obj:`scipy.sparse.linalg.aslinearoperator` coverage.
        It is recommended to use the :meth:`@<__matmul__>` operator or :meth:`dot`
        instead.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__matmul__(x)
    
    def matmat(self, B):
        """Matrix-matrix multiplication ``A B``, implements :meth:`scipy.sparse.linalg.LinearOperator.matmat`.
        
        Note
        ----------
        This method is only here for :obj:`scipy.sparse.linalg.aslinearoperator` coverage.
        It is recommended to use the :meth:`@<__matmul__>` operator or :meth:`dot`
        instead.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.__matmul__(B)
    
    def rmatvec(self, x):
        """Matrix-vector multiplication ``A^T x``, implements :meth:`scipy.sparse.linalg.LinearOperator.rmatvec`.
        
        Note
        ----------
        This method is only here for :obj:`scipy.sparse.linalg.aslinearoperator` coverage.
        It is recommended to use :attr:`A.T <T>` with the :meth:`@<__matmul__>` operator
        or :meth:`dot` instead.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.transpose().__matmul__(x)
        
    def rmatmat(self, B):
        """Matrix-matrix multiplication ``A^T B``, implements :meth:`scipy.sparse.linalg.LinearOperator.rmatmat`.
        
        Note
        ----------
        This method is only here for :obj:`scipy.sparse.linalg.aslinearoperator` coverage.
        It is recommended to use :attr:`A.T <T>` with the :meth:`@<__matmul__>` operator
        or :meth:`dot` instead.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return self.transpose().__matmul__(B)
    
    
    ####################################################
    # other
    ####################################################
    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)
    
class _kronshaped_interface(kron_base):
    """Base class for real Kronecker structured matricies.
    Only addes the Kronecker related attributes :attr:`kshape`.
    """
    
    kdim : int
    r"""Number of concatinated Kronecker products (this is at least 2).
    
    Example: :math:`A \otimes B \otimes C` has ``kdim=3``.
    """
    
    kshape : tuple
    r"""Kronecker shape of the matrix.
    
    Example: :math:`A \otimes B` has 
    ``kshape=((A.shape[0],B.shape[0]),(A.shape[1],B.shape[1]))``,
    i.e. the full :attr:`shape` reads ``shape=tuple(np.prod(kshape,axis=1))``.
    """
    
    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)
        kd = 'kdim=' + str(self.kdim)
        ks = 'kshape=' + str(self.kshape)

        return '<%dx%d %s with %s, %s, %s>' % (M, N, self.__class__.__name__, kd, ks, dt)
        

def iskronrelated(x):
    """Is ``x`` a Kronecker related matrix, i.e. does ``x`` belong to the :mod:`kron` module 
    (namly :obj:`kron_matrix`, :obj:`kronsum_matrix`, :obj:`kronblock_matrix` or :obj:`block_matrix`)?
    
    This methods checks, if ``x`` implements :obj:`kron._interfaces._kron_interface`.
    
    Parameters
    ----------
    x
        object to check for being a Kronecker related matrix
        
    Returns
    -------
    bool
        ``True`` if ``x`` is a Kronecker related matrix, ``False`` otherwise
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return isinstance(x, kron_base)
    
def iskronshaped(x):
    """Is ``x`` a Kronecker shaped matrix, namly :obj:`kron_matrix` or :obj:`kronsum_matrix`?
    
    This methods checks, if ``x`` implements :obj:`kron._interfaces._kronshaped_interface`.
    
    Parameters
    ----------
    x
        object to check for being a Kronecker shaped matrix
        
    Returns
    -------
    bool
        ``True`` if ``x`` is a Kronecker product matrix, ``False`` otherwise
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return isinstance(x, _kronshaped_interface)
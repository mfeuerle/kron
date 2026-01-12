#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Moritz Feuerle, May 2022


__all__ = ['kron_matrix', 'kronsum_matrix', 'kronblock_matrix', 'block_matrix', 'iskron', 'iskronsum', 'iskronblock', 'isblock']


from copy import copy as shallowcopy
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import issparse, coo_array, isspmatrix
from scipy.sparse._sputils import isdense, isscalarlike
import warnings

from kron.utils import iscollection, tosparray, MatrixCollection
from ._interfaces import kron_base, _kronshaped_interface, iskronrelated, iskronshaped




             

class kron_matrix(_kronshaped_interface):
    r"""Multi-layer Kronecker product (or Tensor product) structured matrix
    
    .. math::
        \texttt{A} := \texttt{A[0]} \otimes \texttt{A[1]} \otimes \dots \otimes \texttt{A[kdim-1]}.
    
    The Kronecker product :math:`A \otimes B \in \mathbb{R}^{np \times mq}` of two matricies 
    :math:`A \in \mathbb{R}^{n \times m}` and :math:`B \in \mathbb{R}^{p \times q}` is defined as 
    
    .. math::
        A \otimes B := \begin{pmatrix} a_{11}B & \dots & a_{1m}B \\
                        \vdots & \ddots & \vdots \\
                        a_{n1}B & \dots & a_{nm}B
                        \end{pmatrix}.          
    
    .. todo:: 
        self.dtype verwenden wenn matrix assambled wird oder neue arrays angelegt werden usw.
    .. todo:: 
        eine atomatische withformat, die geeignete formate wählt
    .. todo::
        Index indizierung 
    .. todo::
        Aufteilen in teilblöcke um multiplikationen und addition mit :obj:`block_matrix` zu ermöglichen
        
    """
    
    def __init__(self, *A, copy=False):
        r"""Can be called with a list of matricies or individual matricies
        
        - ``kron_matrix(A1,A2,...)``
        - ``kron_matrix(A1,A2,... , copy=copy)``
        - ``kron_matrix(A)`` with ``A = [A1,A2,...]``
        - ``kron_matrix(A, copy=copy)`` with ``A = [A1,A2,...]``.
        
        Parameters
        ----------
        A
            List of matricies ``A[0],...,A[kdim-1]``.
            Each matrix ``A[i]`` could be either a :obj:`numpy.ndarray`, 
            :obj:`scipy.sparse.sparray` or another :obj:`.kron_matrix`. In the latter case,
            the :obj:`.kron_matrix` gets integrated in the sense that the resulting matrix
            is not a nested Kronecker matrix of Kronecker matricies but a single flat Kronecker matrix 
            of larger Kronecker dimension :attr:`kdim`.
        copy
            If ``True``, the matricies ``A[0],...,A[kdim-1]`` will be copied.
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if len(A) == 1:
            A = A[0]
        assert len(A) >= 2
        
        self._data = self._assemble_data(A)
        if copy:
            self._data = [Ai.copy() for Ai in self._data]
        shapes = np.array([Ai.shape for Ai in self._data]).T
        self.dtype = np.result_type(*[Ai.dtype for Ai in self._data])
        self.kdim = len(self._data)
        self.kshape = tuple(map(tuple, shapes))
        self.shape = tuple(np.prod(shapes, axis=1))
       
       
    def _assemble_data(self,data):
        for A in data:
            if not isdense(A) and not issparse(A) and not iskron(A) and not isscalarlike(A):
                warnings.warn(F"The given input matrix is {type(A)}. The kron_matrix is only tested with "
                              "np.ndarrays, scipy.sparse.sparray and other kron_matrix. "
                              "Be aware of possible errors and inconsistent results.",
                              UserWarning)
                break
        
        data_new = []
        for A in data:
            if iskron(A):
                data_new += A._data
            elif isscalarlike(A):
                data_new += [np.atleast_2d(A)]
            else:
                data_new += [A]
        data = data_new
        
        if any(not np.issubdtype(A.dtype,np.number) and not np.issubdtype(A.dtype,bool) for A in data):
            raise ValueError("matrix is not numeric")
        
        if any(A.ndim != 2 for A in data):
            raise ValueError("only 2 dimensional matricies supported")
        if any(issparse(A) and isspmatrix(A) for A in data):
            warnings.warn("Usage of deprecated scipy.sparse.spmatrix instead of scipy.sparse.sparray; "
                          "elemetnwise multiplication is not supported!", UserWarning)
        return data
        
        
    
    ####################################################
    # element-wise multiplication
    ####################################################
    def __mul__(self,other):        # self * other
        """Elementwise multiplication ``self * other`` of two matricies ``self``
        and ``other`` with same shape, or scalar multiplication where ``other`` is a scalar value.
        
        ``other`` could either be a scalar, another :obj:`kron_matrix` or a :obj:`kronsum_matrix`.
        In the first two cases a new :obj:`kron_matrix`, in the third case a new :obj:`kronsum_matrix` is returned.
        
        Note
        ----------
            This method only handels cases where the Kronecker structure is preserved, the full Kronecker product will
            never be assembled implictly without the users knowlege. Hence, elementwise multiplication with classic 
            matricies like :obj:`scipy.sparse.sparray` or :obj:`numpy.ndarray` is not supported as well as the 
            multiplication with Kronecker matricies with different :attr:`kshape`.
            In this case one hase to manually :meth:`assemble` the matrix first.
        
        Returns
        ----------
        :obj:`kron_matrix` or :obj:`kronsum_matrix`
            Elementwise multiplication of this matrix with ``other``
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
            
        if isscalarlike(other): 
            return kron_matrix(self._data[:-1] + [other*self._data[-1]])
        if iskronshaped(other):
            if iskron(other): 
                return kron_matrix([self._data[i] * other._data[i] for i in range(self.kdim)])
            if iskronsum(other): 
                return other.__rmul__(self) # use rmul from kronsum_matrix
        raise ValueError(f"Elementwise multiplication for {type(other)} not supported, use .assemble()")
    
        
    ####################################################
    # Matrix-Matrix and Matrix-Vector multiplication
    ####################################################
    def __matmul__(self, other):     # self @ other
        r"""Matrix-matrix or matrix-vector multiplication ``self @ other`` of the matrix 
        ``self`` with another matirx or vector ``other``.
        
        In the sense of matrix-vector multiplication ``other`` could either be a dense :obj:`numpy.ndarray` 
        or sparse :obj:`scipy.sparse.sparray`. In the first case the return value will always be a :obj:`numpy.ndarray`,
        in the second it is either :obj:`numpy.ndarray` (if one or more ``A[0],...,A[kdim-1]`` are
        :obj:`numpy.ndarray`) or :obj:`scipy.sparse.sparray` (if all ``A[0],...,A[kdim-1]``
        are :obj:`scipy.sparse.sparray`).
        
        For matrix-matrix multiplication, ``other`` could be another :obj:`kron_matrix` or a :obj:`kronsum_matrix`, then
        the return value will be a :obj:`kron_matrix` or :obj:`kronsum_matrix`, respectively.
        
        Note
        ----------
            - Of course one could also multiply dense and sparse matricies, not only
              vectors, but this is only recommended for small matricies.
            - The full Kronecker product will never be assembled implictly without the users knowlege. 
              Hence, the matrix-matrix multiplication with another :obj:`kron_matrix` or a :obj:`kronsum_matrix`
              is only for compatible :attr:`kshape` supported, where the Kronecker structure could be
              preserved. In other cases one hase to manually :meth:`assemble` the matrix first.
        
        Returns
        ----------
        :obj:`kron_matrix`, :obj:`kronsum_matrix`, :obj:`scipy.sparse.sparray` or :obj:`numpy.ndarray`
            Matrix-matrix / matrix-vector multiplication of this matrix with ``other``.
          
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"dimension missmatch between {self.shape} and {other.shape}")
        if iskronshaped(other):
            if self.kshape[1] != other.kshape[0]:
                raise ValueError(f"kron dimension missmatch {self.kshape} and {other.kshape}, use .asformat()")
            if iskron(other):
                # (A kron B) * (C kron D) = (AC kron BD)
                return kron_matrix([self._data[i] @ other._data[i] for i in range(self.kdim)])
            if iskronsum(other): 
                # use function from kronsum_matrix
                return other.__rmatmul__(self)
        if isdense(other):
            return self._matmul_vector_dense(other)
        if issparse(other):
            return self._matmul_vector_sparse(other)
        raise ValueError(f"Matrix multiplication for {type(other)} not supported, use .assemble()")
    
    
    def __rmatmul__(self, other):       # other @ self
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        if other.shape[1] != self.shape[0]:
            raise ValueError(f"dimension missmatch between {other.shape} and {self.shape}")
        if isdense(other):
            # more efficient implementation possible
            return self.transpose()._matmul_vector_dense(other.transpose()).transpose()
        if issparse(other):
            # more efficient implementation possible
            return self.transpose()._matmul_vector_sparse(other.transpose()).transpose()
        if iskronshaped(other):
            if other.kshape[1] != self.kshape[0]:
                raise ValueError(f"kron dimension missmatch {other.kshape} and {self.kshape}, use .asformat()")
            if iskron(other):
                # (A kron B) * (C kron D) = (AC kron BD)
                return kron_matrix([other._data[i] @ self._data[i] for i in range(self.kdim)])
            if iskronsum(other):
                # use function from kronsum_matrix
                return other.__matmul__(self)
        raise ValueError(f"Matrix multiplication for {type(other)} not supported, use .assemble()")
        
        
    def _matmul_vector_dense(self, x):
        """ Memory efficient matrix-vector multiplication with a dense vector, where we make use of 
        ( A kron B ) * x = Vector( B * Matrix(x) * A^T ) 
        to avoid assembeling the matrix.
        
        .. todo:: analoge implementierung für :meth:`__rmatmul__` um überflüssiges transponieren zu vermeiden
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        assert(1 <= x.ndim <= 2)        
        if self.kdim > 2:
            A = kron_matrix(self._data[:-1])
        else:
            A = self._data[0]
        B = self._data[-1].transpose()
        
        if x.ndim == 1:
            ABx = A @ x.reshape((A.shape[1],B.shape[0])) @ B
            return ABx.reshape((-1))
        
        x = x.reshape((A.shape[1],B.shape[0],-1))
        ABx = np.zeros((A.shape[0],B.shape[1],x.shape[2]))
        for i in range(x.shape[2]):
            ABx[:,:,i] = A @ x[:,:,i] @ B
    
        return ABx.reshape((A.shape[0]*B.shape[1], -1))
    
    def _matmul_vector_sparse(self, x, rmatmul=False):
        """Memory efficient matrix-vector multiplication with a sparse vector, where we make use of 
        ( A kron B ) * x = Vector( B * Matrix(x) * A^T ) 
        to avoid assembeling the matrix.
        
        .. todo:: analoge implementierung für :meth:`__rmatmul__` um überflüssiges transponieren zu vermeiden
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.kdim > 2:
            A = kron_matrix(self._data[:-1])
        else:
            A = self._data[0]
        B = self._data[-1].transpose()
            
        x = x.tocsc()
        ABx = []
        for i in range(x.shape[1]):
            xi = x[:,i].reshape((A.shape[1],B.shape[0]))
            ABx.append( (A @ xi @ B).reshape((A.shape[0]*B.shape[1],1)) )
        if isdense(ABx[0]):
            return np.block(ABx)
        else:
            return tosparray(sparse.hstack(ABx))
    
    
    ####################################################
    # Addition
    ####################################################
    def __add__(self, other):   # self + other
        """Addition ``self + other``.
        If ``other`` is not zero, the addition is directly passed to :meth:`kronsum_matrix.__add__`.
        For more informations on compatibility terms, have a look there.
        
        Returns
        ----------
        :obj:`kronsum_matrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        if isscalarlike(other) and other == 0:
            return shallowcopy(self)
        return kronsum_matrix([self]) + other
    
    
    def sum(self, axis=None):
         if axis is None:
            return np.prod([A.sum() for A in self._data])
         else:
            data = self.kdim * [None]
            for k in range(self.kdim):
                data[k] = self._data[k].sum(axis=axis)
                if data[k].ndim == 1:
                    shape = list(self._data[k].shape)
                    shape[axis] = 1
                    data[k] = data[k].reshape(shape)
                    
            return kron_matrix(data)
            
    ####################################################
    # Adjoint matrix
    ####################################################
    def transpose(self):
        r"""Returnes the transposed matrix ``A^T``.
            
        Returns
        ----------
        :obj:`kron_matrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return kron_matrix([A.transpose() for A in self._data])
    
    ####################################################
    # absolut value
    ####################################################
    def __abs__(self):
        return kron_matrix([abs(A) for A in self._data])
    
    ####################################################
    # access kronecker layers
    ####################################################
    def getklayer(self, k):
        """Returnes the matrix of the k-th Kronecker layer ``A[k]``.
        
        Parameters
        ----------
        k
            Kronecker layer (``0 <= k <`` :attr:`kdim`)
        
        Returns
        ----------
        :obj:`numpy.ndarray` or :obj:`scipy.sparse.sparray`
            ``A[k]``
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        assert k in range(self.kdim)
        return self._data[k]
    
    
    ####################################################
    # formation export
    ####################################################
    def asformat(self, format, copy=True):          
        if format == 'array' or format == 'dense':
            K = self._data[0] if isdense(self._data[0]) else self._data[0].asformat(format)
            for i in range(1,self.kdim):
                K = np.kron(K, self._data[i] if isdense(self._data[i]) else self._data[i].asformat(format))
        else:
            if isdense(self._data[0]):
                K = coo_array(self._data[0])
            elif iskronrelated(self._data[0]):
                K = self._data[0].assemble()
            else:
                K = self._data[0].tocoo()
            for i in range(1,self.kdim):
                if isdense(self._data[i]):
                    K = sparse.kron(K, coo_array(self._data[i]))
                elif iskronrelated(self._data[i]):
                    K = sparse.kron(K, self._data[i].assemble())
                else:
                    K = sparse.kron(K, self._data[i].tocoo())
            K = tosparray(K)
            if format is not None:
                K = K.asformat(format=format)
        return K
    
    @property
    def format(self):
        r"""List of the formats of the individual matricies ``A[0],...,A[kdim-1]``.
        
        ``'dense'`` for :obj:`numpy.ndarray`, :attr:`scipy.sparse.sparray.format` for :obj:`scipy.sparse.sparray`
        or otherwise the string reprensentation of the data type.
        
        Returns
        ----------
        List of strings
            Formats of ``A[0],...,A[kdim-1]``:
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        format = []
        for A in self._data:
            if isdense(A): 
                format.append('dense')
            else:
                try:
                    format.append(A.format)
                except:
                    format.append(str(type(A)))
        return format
    
    def withformat(self, format, copy=False):
        r"""Changes the internal format of the individual matricies ``A[0],...,A[kdim-1]``.
        For a list of formats see ``asformat``.

        Parameters
        ----------
        format
            Eiter a single format, to witch all matricies are converted,
            or a list of formats, one for every ``A[0],...,A[kdim-1]``.
        copy
            If ``False``, this is a inplace operation and changes this matrix. If ``True``,
            a new matrix with the given  formats is returned, while this matrix stays unchanged.

        Returns
        -------
        :obj:`kron_matrix`
            Either ``self`` if ``copy=False`` or a new matrix with the given formats.
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        
        if isinstance(format, str):
            format = [format for A in self._data]
        if len(format) != self.kdim:
            raise ValueError()
        
        if copy:
            data = []
            for i in range(self.kdim):
                if format[i] == 'array' or format[i] == 'dense':
                    data.append(self._data[i].copy() if isdense(self._data[i]) else self._data[i].asformat(format[i],copy=True))
                else:
                    data.append((coo_array(self._data[i]) if isdense(self._data[i]) else self._data[i]).asformat(format[i],copy=True))
            return kron_matrix(data)
        else:
            for i in range(self.kdim):
                if format[i] == 'array' or format[i] == 'dense':
                    self._data[i] = self._data[i] if isdense(self._data[i]) else self._data[i].asformat(format[i],copy=False)
                else:
                    self._data[i] = (coo_array(self._data[i]) if isdense(self._data[i]) else self._data[i]).asformat(format[i],copy=False)
            return self

        
    def copy(self):
        """Copys the matrix and all internal data such that the old and the new matrix do not share any data.

        Returns
        -------
        :obj:`kron_matrix`
            copy of this matrix
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return kron_matrix([A.copy() for A in self._data])
        
    def diagonal(self):
        """Returns the diagonal of the matrix.

        Returns
        -------
        :obj:`kron_matrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.kshape[0] != self.kshape[1]:
            return NotImplementedError("right now only squared matricies supported")
        return kron_matrix([A.diagonal().reshape(1,-1) for A in self._data])

    
    def eliminate_zeros(self):
        r"""Eliminates zero entrys in the individual matricies ``A[0],...,A[kdim-1]`` 
        (unless they are :obj:`numpy.ndarray`).
        
        Warning
        ----------
        This operation could not only affect this matrix, but also other 
        matricies that share the same internal matricies ``A[0],...,A[kdim-1]``. 
        Nevertheless, this should not leed to any critical side effects, since only zeros are removed.
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        for i in range(self.kdim):
            if hasattr(self._data[i], 'eliminate_zeros'):
                self._data[i].eliminate_zeros()
        
                                               
class kronsum_matrix(_kronshaped_interface):
    r"""Summation of several :obj:`kron_matrix`. ``A[0] + A[1] +  + A[N-1]``.

    The matricies ``A[0],...,A[N-1]`` have to have the same :attr:`kshape`. 
    Internally, the matrix sum is never calculated and each matrix is stored as is. 
    This is usefull for a sum of 
    :obj:`kron_matrix` to preserve the Kronecker structure in memory, since the Kronecker 
    structure is in general lost during summation.        
        
    .. todo:: 
        self.dtype verwenden wenn matrix assambled wird oder neue arrays angelegt werden usw. 
    """
    
    N : int
    """Number of summed matricies"""
    
    def __init__(self, A, copy=False):
        r"""
        Parameters
        ----------
        A
            List of matricies ``A[0],...,A[N-1]``.
            Each matrix ``A[i]`` could be either a :obj:`.kron_matrix` or :obj:`.kronsum_matrix`. 
            In the latter case, the :obj:`.kronsum_matrix` gets integrated in the sense that the resulting matrix
            is not a nested sum of several summs but a single flat sum.
        copy
            If ``True``, the matricies ``A[0],...,A[N-1]`` will be copied.
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        assert len(A)>0
        assert all(iskronshaped(Ai)  for Ai in A)
        assert all(A[0].shape == Ai.shape for Ai in A)
        assert all(A[0].kshape == Ai.kshape for Ai in A)
        
        self._data = self._assemble_data(A)
        if copy:
            self._data = [Ai.copy() for Ai in self._data]
        self.dtype = np.result_type(*[Ai.dtype for Ai in self._data])
        self.shape = self._data[0].shape
        self.kshape = self._data[0].kshape
        self.kdim = self._data[0].kdim
        self.N = len(self._data)
        
    def _assemble_data(self,data):
        for A in data:
            if not iskron(A) and not iskronsum(A):
                warnings.warn(F"The given input matrix is {type(A)}. The kronsum_matrix is only tested with kron_matrix. "
                              "Be aware of possible errors and inconsistent results.",
                              UserWarning)
                break
        
        data = sum([A._data if iskronsum(A) else [A] for A in data],[])
        return data
        
    
    ####################################################
    # element-wise multiplication
    ####################################################    
    def __mul__(self,other):        # self * other
        """Elementwise multiplication ``self * other`` of two matricies ``self``
        and ``other`` with same shape, or scalar multiplication where ``other`` is a scalar value.
        
        Since ``other`` is multiplyed with all ``A[i]``, it has to be compatible with 
        :meth:`A[i] * other <kron_matrix.__mul__>`.
        
        Returns
        ----------
        :obj:`kronsum_matrix`
            Elementwise multiplication of this matrix with ``other``
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        return sum([A * other for A in self._data])
    
    def __rmul__(self,other):       # other * self
        """Elementwise multiplication ``other * self`` of two matricies ``self``
        and ``other`` with same shape, or scalar multiplication where ``other`` is a scalar value.
        
        Since ``other`` is multiplyed with all ``A[i]``, it has to be compatible with 
        :meth:`other * A[i] <kron_matrix.__rmul__>`.
        
        Returns
        ----------
        :obj:`kronsum_matrix`
            Elementwise multiplication of this matrix with ``other``
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return sum([other * A for A in self._data])
    
    ####################################################
    # Matrix-Matrix and Matrix-Vector multiplication
    ####################################################   
    def __matmul__(self,other):     # self @ other
        """Matrix-matrix or matrix-vector multiplication ``self @ other`` of the matrix 
        ``self`` with another matirx or vector ``other``.
        
        Since ``other`` is multiplyed with all ``A[i]``, it has to be compatible with 
        :meth:`A[i] @ other <kron_matrix.__matmul__>`.
        
        The results of :meth:`A[i] @ other <kron_matrix.__matmul__>` will all be calculated and added together. 
        Depending on the return types of :meth:`A[i] @ other <kron_matrix.__matmul__>`, the resulting
        sum is either a :obj:`kronsum_matrix`, a :obj:`scipy.sparse.sparray` or a :obj:`numpy.ndarray`.
        
        In short: If ``other`` is :obj:`numpy.ndarray` a :obj:`numpy.ndarray` will be returned.
        If ``other`` is :obj:`scipy.sparse.sparray` a :obj:`scipy.sparse.sparray` or :obj:`numpy.ndarray`
        will be returned. If ``other`` is :obj:`kron_matrix` or :obj:`kronsum_matrix` a :obj:`kronsum_matrix`
        will be returned.
        
        Returns
        ----------
        :obj:`kronsum_matrix`, :obj:`scipy.sparse.sparray` or :obj:`numpy.ndarray`
            Matrix-matrix / matrix-vector multiplication of this matrix with ``other``
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        return sum([A @ other for A in self._data])
    
    def __rmatmul__(self,other):    # other @ self
        return sum([other @ A for A in self._data])
    
    ####################################################
    # Addition
    ####################################################
    def __add__(self, other):       # self + other
        r"""Addition ``self + other``.
        
        Aside from zero, ``other`` can be a :obj:`kron_matrix` or :obj:`kronsum_matrix` of 
        the same :attr:`kshape`. In the first case, ``other`` is just appended to summation list
        ``A[0],...,A[N-1]``. In the second case, the two summation lists of ``self`` and
        ``other`` are concatenated.

        Returns
        ----------
        :obj:`kronsum_matrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        if isscalarlike(other):
            if other == 0:
                return shallowcopy(self)
            raise ValueError("adding a nonzero scalar to a kron matrix is not supported")
        if self.shape != other.shape:
            raise ValueError(f"dimension missmatch between {self.shape} and {other.shape}")
        if iskronshaped(other):
            if self.kshape != other.kshape:
                raise ValueError(f"kron dimension missmatch {self.kshape} and {other.kshape}, use .assemble()")
            if iskronsum(other):
                return kronsum_matrix(self._data + other._data)
            if iskron(other):
                return kronsum_matrix(self._data + [other])
        raise ValueError(f"Addition of {type(other)} not supported, use .assemble()")
    
    
    def sum(self, axis=None):
        return sum(A.sum(axis=axis) for A in self._data)

    ####################################################
    # Adjoint matrix
    ####################################################
    def transpose(self):
        r"""Returnes the transposed matrix ``A^T``.
            
        Returns
        ----------
        :obj:`kronsum_matrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return kronsum_matrix([A.transpose() for A in self._data])
    
    ####################################################
    # absolut value
    ####################################################
    def __abs__(self):
        raise RuntimeError("abs for kronsum not possible, use .assemble()")
    
    ####################################################
    # access kronecker layers
    ####################################################
    def getklayer(self, k):
        """Returnes the matricies of the k-th Kronecker layer of each ``A[0],...,A[N-1]``.
        
        Parameters
        ----------
        k
            Kronecker layer (``0 <= k <`` :attr:`kdim`)
        
        Returns
        ----------
        :obj:`utils.MatrixCollection`
            
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        assert k in range(self.kdim)
        return MatrixCollection([A.getklayer(k) for A in self._data])
    
    ####################################################
    # formation export
    ####################################################
    def asformat(self, format, copy=True):         
        if format == 'array' or format == 'dense':
            A = self._data[0] if isdense(self._data[0]) else self._data[0].asformat(format)
            for i in range(1,self.N):
                if isdense(self._data[i]):
                    A += self._data[i]
                else:
                    A += self._data[i].asformat(format)
        else:
            A = 0
            for i in range(self.N):
                if isdense(self._data[i]):
                    A += coo_array(self._data[i])
                elif iskronrelated(self._data[i]):
                    A += self._data[i].assemble()
                elif issparse(self._data[i]):
                    A += self._data[i]
                else:
                    A += self._data[i].tocoo()
            if format is not None:
                A = A.asformat(format=format)
        return A
    
    ####################################################
    # other
    ####################################################
    
    def copy(self):
        """Copys the matrix and all internal data such that the old and the new matrix do not share any data.

        Returns
        -------
        :obj:`kron_matrix`
            copy of this matrix
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        return kronsum_matrix([A.copy() for A in self._data])
    
    def diagonal(self):
        """Returns the diagonal of the matrix.

        Returns
        -------
        :obj:`kronsum_matrix`
        """
        return kronsum_matrix([A.diagonal() for A in self._data])
    
    def eliminate_zeros(self):
        r"""Eliminates zero entrys in the individual matricies ``A[0],...,A[N-1]``.
        
        Warning
        ----------
        This operation could not only affect this matrix, but also other 
        matricies that share the same internal matricies ``A[0],...,A[N-1]``. 
        Nevertheless, this should not leed to any critical side effects, since only zeros are removed.
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        for i in range(self.N):
            self._data[i].eliminate_zeros()
            
            
    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)
        kd = 'kdim=' + str(self.kdim)
        ks = 'kshape=' + str(self.kshape)
        n = 'N=' + str(self.N)

        return '<%dx%d %s with %s, %s, %s, %s>' % (M, N, self.__class__.__name__, n, kd, ks,dt)

            
class block_matrix(kron_base):
    r"""Block matrix

    .. math::
        A = \begin{pmatrix} \texttt{blocks[0][0]} & \dots & \texttt{blocks[0][N-1]} \\
            \vdots & \ddots & \vdots \\
            \texttt{blocks[M-1][0]} & \dots & \texttt{blocks[M-1][N-1]}
            \end{pmatrix}.
    
    of matrices with compatible shapes.

    The blocks are internally stored as is, i.e. the block matrix is not 
    assembled in memory. This is usefull for Kronecker shaped 
    matricies, since the Kronecker structure of each subblock would be lost by
    assebeling the whole matrix, resulting in a much larger memory consumption.        
        
    .. todo:: 
        self.dtype verwenden wenn matrix assambled wird oder neue arrays angelegt werden usw. 
    .. todo::
        sinnvoll die None durch kron_matrix bestehend aus leeren coo_arrays zu ersetzen?
    .. todo:: 
        sinnvoll sich auf Kronecker shaped blöcke zu beschränken, also in _assemble_block_shape auch kshape checken?
    .. todo::
        sollten blöcke von blöcken beim init auch integriert werden in eine block struktur?
    """
    
    bshape : tuple
    """Block shape ``bshape=(M,N)``"""
    
    sbshape : tuple
    """Sub-block shapes ``blocks[i][j].shape = (sbshape[0][i], sbshape[1][j])``"""
    
    def __init__(self, blocks, copy=False):
        """
        Parameters
        ----------
        blocks : array_like
            Grid of matrices with compatible shapes, has to be 2D. An entry of ``None`` implies an all-zero sub-block.
        copy
            If ``True``, the matricies in ``blocks`` will be copied.
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """        
        
        blocks, dtype, sbshape= self._assemble_block_shape(blocks,copy)
        
        self._blocks  = blocks
        self._offsets = [np.append(0, np.cumsum(sbshape[i])) for i in range(2)]
        
        self.dtype = dtype
        self.shape = tuple(self._offsets[i][-1] for i in range(2))
        self.bshape = blocks.shape
        self.sbshape = sbshape
        self.kdim = 1
        
        
    def _assemble_block_shape(self,blocks,copy):      
        M = len(blocks)
        try:
            N = len(blocks[0])
        except TypeError:
            raise ValueError('blocks must be 2D')
            
        brow_lengths = np.zeros(M, dtype=np.int64)
        bcol_lengths = np.zeros(N, dtype=np.int64)
        
        dtype = []
        
        for i in range(M):
            if len(blocks[i]) != N:
                raise ValueError(f"inconsisten block shape, expectet blocks.shape={(M,N)}, but len(blocks[{i}]) = {len(blocks[i])}")
            for j in range(N):
                if blocks[i][j] is not None:
                    A = blocks[i][j]
                    if A.ndim != 2:
                        raise ValueError("Only 2 dimensional matricies supported")
                    
                    dtype.append(A.dtype)
                    
                    if brow_lengths[i] == 0:
                        brow_lengths[i] = A.shape[0]
                    elif brow_lengths[i] != A.shape[0]:
                        msg = (f'blocks[{i},:] has incompatible row dimensions. '
                            f'Got blocks[{i},{j}].shape[0] == {A.shape[0]}, '
                            f'expected {brow_lengths[i]}.')
                        raise ValueError(msg)

                    if bcol_lengths[j] == 0:
                        bcol_lengths[j] = A.shape[1]
                    elif bcol_lengths[j] != A.shape[1]:
                        msg = (f'blocks[:,{j}] has incompatible column '
                            f'dimensions. '
                            f'Got blocks[{i},{j}].shape[1] == {A.shape[1]}, '
                            f'expected {bcol_lengths[j]}.')
                        raise ValueError(msg)
                    
        sbshape = tuple(brow_lengths), tuple(bcol_lengths)
        dtype = np.result_type(*dtype)
        blocks_new = np.full((M,N), fill_value=None, dtype='object')
        for i in range(M):
            for j in range(N):
                if blocks[i][j] is None:
                    blocks_new[i,j] = coo_array((sbshape[0][i],sbshape[1][j]), dtype=dtype)
                else:
                    blocks_new[i,j] = blocks[i][j].copy() if copy else blocks[i][j]
        
        return blocks_new, dtype, sbshape
            
    
    ####################################################
    # element-wise multiplication
    ####################################################    
    def __mul__(self, other):
        """Elementwise multiplication ``self * other`` of two matricies ``self``
        and ``other`` with same shape, or scalar multiplication where ``other`` is a scalar value.
        
        ``other`` could either be a scalar, :obj:`scipy.sparse.sparray`, :obj:`numpy.ndarray`
        or another :obj:`block_matrix` withe the same :attr:`sbshape`.
        
        Returns
        ----------
        :obj:`block_matrix`
            Elementwise multiplication of this matrix with ``other``
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        if isscalarlike(other): 
            return self._mul_scalar(other)
        if isblock(other):     
            return self._mul_block_matrix(other)
        if iskronshaped(other):
            return self._mul_kron_interface(other)
        if isdense(other) or issparse(other):
            return self._mul_default(other)
        raise ValueError(f"Elementwise multiplication for {type(other)} not supported, use .assemble()")
    
    
    def _mul_scalar(self, alpha):
        return self.__class__(alpha * self._blocks)
    
    def _mul_block_matrix(self, other):
        return self.__class__(self._blocks * other._blocks)
    
    def _mul_kron_interface(self, other):
        # return self._mul_block_matrix(other.blocks(sbshape=self.sbshape))
        raise NotImplementedError("multiplication of block matrix and unblocked matrix currently not supported, use .assemble().")
        
    def _mul_default(self, other):
        if other.ndim == 1:
            if other.shape[0] != self.shape[1]:
                raise ValueError("dimension missmatch")
            other_blocks = np.empty(shape=(self.bshape[1],), dtype='object')
            for i in range(self.bshape[1]):
                i_idx = slice(self._offsets[1][i], self._offsets[1][i+1])
                other_blocks[i] = other[i_idx]
        else:
            if (self.shape[0] != other.shape[0] and 1 != other.shape[0]) or (self.shape[1] != other.shape[1] and 1 != other.shape[1]):
                raise ValueError("dimension missmatch")
            other_blocks = np.empty(shape=self.bshape, dtype='object')
            for i in range(self.bshape[0]):
                idx_i = [0] if other.shape[0]==1 else slice(self._offsets[0][i], self._offsets[0][i+1])
                for j in range(self.bshape[1]):
                    idx_j = [0] if other.shape[1]==1 else slice(self._offsets[1][j], self._offsets[1][j+1])
                    other_blocks[i,j] = other[idx_i,:][:,idx_j]
        return self.__class__(self._blocks * other_blocks)

            
    ####################################################
    # Matrix-Matrix and Matrix-Vector multiplication
    ####################################################        
    def __matmul__(self, other):
        r"""Matrix-matrix or matrix-vector multiplication ``self @ other`` of the matrix 
        ``self`` with another matirx or vector ``other``.
        
        In the sense of matrix-vector multiplication ``other`` could either be a dense :obj:`numpy.ndarray` 
        or sparse :obj:`scipy.sparse.sparray`. In the first case the return value will always be a :obj:`numpy.ndarray`,
        in the second it is either :obj:`numpy.ndarray` or :obj:`scipy.sparse.sparray`.
        
        For matrix-matrix multiplication, ``other`` could be another block matrix with compatible :attr:`sbshape`,
        then the returned value will be a block matrix.
        
        Returns
        ----------
        :obj:`block_matrix`, :obj:`kronblock_matrix`, :obj:`scipy.sparse.sparray` or :obj:`numpy.ndarray`
            Matrix-matrix / matrix-vector multiplication of this matrix with ``other``
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        assert other.ndim <= 2
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"dimension missmatch between {self.shape} and {other.shape}")
        if isblock(other):    
            return self._matmul_block_matrix(other)
        if iskronshaped(other):
            return self._matmul_kron_interface(other)
        if isdense(other):
            return self._matmul_vector_dense(other)
        if issparse(other):
            return self._matmul_vector_sparse(other)
        raise ValueError(f"Matrix multiplication for {type(other)} not supported, use .assemble()")
    
    def __rmatmul__(self, other):
        assert other.ndim <= 2
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        if other.shape[-1] != self.shape[0]:
            raise ValueError(f"dimension missmatch between {other.shape} and {self.shape}")
        if isdense(other):
            return self._rmatmul_vector_dense(other)
        if issparse(other):
            return self._rmatmul_vector_sparse(other)
        if iskronshaped(other):
            return self._rmatmul_kron_interface(other)
        if isblock(other):
            return other._matmul_block_matrix(self)
        raise ValueError(f"Matrix multiplication for {type(other)} not supported, use .assemble()")
    
    
    def _matmul_block_matrix(self, other):           
        if self.sbshape[1] != other.sbshape[0]:
            raise ValueError(f"block dimension missmatch {self.sbshape} and {other.sbshape}. "
                                      "Block-resizing not supported, use .asformat()")

        blocks = np.full(shape=(self.bshape[0],other.bshape[1]), fill_value=0, dtype='object')
        for i in range(self.bshape[0]):
            for k in range(self.bshape[1]):
                for j in range(other.bshape[1]):
                    blocks[i,j] += self._blocks[i,k] @ other._blocks[k,j]
        return self.__class__(blocks)
        
        
    def _matmul_kron_interface(self, other):
        # return self._matmul_block_matrix(other.blocks(sbshape=(self.sbshape[1],-1)))
        raise NotImplementedError("matrix multiplication of block matrix and unblocked matrix currently not supported, use .assemble().")
    
    def _rmatmul_kron_interface(self, other):
        raise NotImplementedError("matrix multiplication of block matrix and unblocked matrix currently not supported, use .assemble().")
    
    
    def _matmul_vector_dense(self, x):
        ndim1 = False
        if x.ndim == 1:
            ndim1 = True
            x = x.reshape((-1,1))
        Ax = np.zeros(shape=(self.shape[0],x.shape[1]), dtype=self.dtype)       
        for j in range(self.bshape[1]):
            j_idx = slice(self._offsets[1][j], self._offsets[1][j+1])
            xj = x[j_idx,:]
            for i in range(self.bshape[0]):
                i_idx = slice(self._offsets[0][i], self._offsets[0][i+1])
                Ax[i_idx,:] += self._blocks[i,j] @ xj
        if ndim1:
            Ax = Ax.reshape((-1,))
        return Ax
        
    def _rmatmul_vector_dense(self, x):
        ndim1 = False
        if x.ndim == 1:
            ndim1 = True
            x = x.reshape((1,-1))
        xA = np.zeros(shape=(x.shape[0],self.shape[1]), dtype=self.dtype)       
        for i in range(self.bshape[0]):
            i_idx = slice(self._offsets[0][i], self._offsets[0][i+1])
            xi = x[:,i_idx]
            for j in range(self.bshape[1]):
                j_idx = slice(self._offsets[1][j], self._offsets[1][j+1])
                xA[:,j_idx] += xi @ self._blocks[i,j]
        if ndim1:
            xA = xA.reshape((-1,))
        return xA
            
            
    def _matmul_vector_sparse(self, x):
        Ax = self.bshape[0] * [0]
        for j in range(self.bshape[1]):
            x_j = x[self._offsets[1][j]:self._offsets[1][j+1], :]
            for i in range(self.bshape[0]):
                Ax[i] += self._blocks[i][j] @ x_j
                if isdense(Ax[i]): 
                    Ax[i] = x.__class__(Ax[i])
        if self.bshape[0] == 1:
            return Ax[0]
        else:
            return tosparray(sparse.vstack(Ax))
        
    def _rmatmul_vector_sparse(self, x):
        xA = self.bshape[1] * [0]
        for i in range(self.bshape[0]):
            x_i = x[:, self._offsets[0][i]:self._offsets[0][i+1]]
            for j in range(self.bshape[1]):
                xA[j] += x_i @ self._blocks[i][j]
                if isdense(xA[j]): 
                    xA[j] = x.__class__(xA[j])
        if self.bshape[1] == 1:
            return xA[0]
        else:
            return tosparray(sparse.hstack(xA))
                 
        
    ####################################################
    # Addition
    ####################################################
    def __add__(self, other):
        """Addition ``self + other``.
        
        Aside from zero, only other :obj:`block_matrix` withe the same :attr:`sbshape` are supported 
        for types for ``other``.
        In other cases one hase to manually :meth:`assemble` the matrix first.

        Returns
        ----------
        :obj:`block_matrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.__kron_priority__ < getattr(other, "__kron_priority__", -1.0):
            return NotImplemented
        
        if isscalarlike(other):
            if other == 0:
                return shallowcopy(self)
            raise ValueError("adding a nonzero scalar to a block_matrix is not supported")
        if self.shape != other.shape:
            raise ValueError(f"dimension missmatch between {self.shape} and {other.shape}") 
        if isblock(other):     
            return self._add_block_matrix(other)
        if iskronshaped(other):
            return self._add_kron_interface(other)
        if isdense(other) or issparse(other):
            return self._add_default(other)
        raise ValueError(f"Addition of {type(other)} not supported, use .assemble()")
        
    def _add_block_matrix(self, other):
        if self.sbshape != other.sbshape:
            raise ValueError(f"different block sizes {self.sbshape} and {other.sbshape}."
                              "block changes currently not supported, use .assemble().")
        blocks = np.empty(shape=self.bshape, dtype='object')
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                blocks[i,j] = self._blocks[i,j] + other._blocks[i,j]
        return self.__class__(blocks) 
            
    def _add_kron_interface(self, other):
        # return self._add_block_matrix(other.blocks(sbshape=self.sbshape))
        raise NotImplementedError("addition of block matrix and unblocked kron matrix currently not supported, use .assemble().")   
    
    def _add_default(self, other):
        blocks = np.empty(shape=self.bshape, dtype='object')
        for i in range(self.bshape[0]):
            col_idx = slice(self._offsets[0][i],self._offsets[0][i+1])
            for j in range(self.bshape[1]):
                row_idx = slice(self._offsets[1][j],self._offsets[1][j+1])
                blocks[i,j] = self._blocks[i,j] + other[col_idx,:][:,row_idx]
        return self.__class__(blocks)
                

    def sum(self, axis=None):
        M,N = self.bshape
        if axis is None:
            out = 0
            for i in range(M):
                for j in range(N):
                    out += self._blocks[i,j].sum()
            return out
        else:
            blocks = np.empty(shape=(M,N), dtype='object')
            for i in range(M):
                for j in range(N):
                    blk = self._blocks[i,j]
                    blocks[i,j] = blk.sum(axis=axis)
                    if blocks[i,j].ndim == 1:
                        shape = list(blk.shape)
                        shape[axis] = 1
                        blocks[i,j] = blocks[i,j].reshape(shape)
            return self.__class__(blocks.sum(axis=axis, keepdims=True))
            
        
    ####################################################
    # Adjoint matrix
    ####################################################        
    def transpose(self):      
        """Returnes the transposed matrix ``A^T``.
            
        Returns
        ----------
        :obj:`block_matrix`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """  
        blocks = np.empty(shape=(self.bshape[1],self.bshape[0]), dtype='object')
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                blocks[j,i] = self._blocks[i,j].transpose()
        return self.__class__(blocks)
    
    
    ####################################################
    # absolut value
    ####################################################
    def __abs__(self):
        blocks = np.empty(self.bshape, dtype='object')
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                blocks[i,j] = abs(self._blocks[i,j])
        return self.__class__(blocks)
    
     
    ####################################################
    # access kronecker layers
    ####################################################
    def getklayer(self, k):
        """Returnes the matricies of the k-th Kronecker layer of each sub-block.
        This is not a real Kronecker matrix with ``kdim=1``, i.e. ``k=0`` is the 
        only layer, and it will return just ``blocks``.  
        
        Parameters
        ----------
        k
            Kronecker layer (``0 <= k <`` :attr:`kdim`)
        
        Returns
        ----------
        :obj:`block_matrix`
            resturns self, sinc this is the only layer
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        assert k in range(self.kdim)
        return shallowcopy(self)
     
    
    ####################################################
    # formation export
    ####################################################
    def asformat(self, format, copy=True):        
        blocks = np.empty(shape=self.bshape, dtype='object')
        if format == 'array' or format == 'dense':
            for i in range(self.bshape[0]):
                for j in range(self.bshape[1]):
                    if isdense(self._blocks[i,j]):
                        blocks[i,j] = self._blocks[i,j]
                    else:
                        blocks[i,j] = self._blocks[i,j].asformat(format)
            return np.block(blocks.tolist())
        else:
            for i in range(self.bshape[0]):
                for j in range(self.bshape[1]):
                    if isdense(self._blocks[i,j]):
                        blocks[i,j] = coo_array(self._blocks[i,j])    
                    elif iskronrelated(self._blocks[i,j]):
                        blocks[i,j] = self._blocks[i,j].assemble()
                    else:
                        blocks[i,j] = self._blocks[i,j]
            M = tosparray(sparse.bmat(blocks.tolist()))
            if format is not None:
                M = M.asformat(format=format)
            return M
        
        
    
    ####################################################
    # Matrix collections
    ####################################################
    def hascollections(self):
        """Checks if any of the subblocks is a :obj:`utils.MatrixCollection`.

        Returns
        -------
        bool
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        M,N = self.bshape
        for i in range(M):
            for j in range(N):
                if iscollection(self._blocks[i,j]):
                    return True
        return False
    
    def iscollection(self):
        """Checks if the matrix is a collection of block matricies, i.e. if all
        blocks are :obj:`utils.MatrixCollection` with same length.

        Returns
        -------
        int
            0 if not, commen length of the collections in each block otherwise
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        M,N = self.bshape
        L = -1
        for i in range(M):
            for j in range(N):
                if iscollection(self._blocks[i,j]):
                    if L == -1:
                        L = len(self._blocks[i,j])
                    elif L != len(self._blocks[i,j]):
                        return 0
                else:
                    return 0
        return L
           
    def ascollection(self):
        """If this is a block matrix, where each subblock is a :obj:`utils.MatrixCollection`, all having the same length,
        the block matrix of collections will be converted to a :obj:`utils.MatrixCollection` where each element is a block matrix.
        

        Returns
        -------
        :obj:`utils.MatrixCollection`
            matrix collection where each entry is block matrix

        Raises
        ------
        ValueError
            can not be converted to :obj:`utils.MatrixCollection`, see :meth:`iscollection`
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        M,N = self.bshape
        L = self.iscollection()
        if not L:
            raise ValueError("is no collection")
        collection = []
        for l in range(L):
            blocks = np.empty(shape=(M,N), dtype='object')
            for i in range(M):
                for j in range(N):
                    blocks[i,j] = self._blocks[i,j][l]
            collection.append(self.__class__(blocks))
        return MatrixCollection(collection)
    
    ####################################################
    # other
    ####################################################
    def copy(self):
        """Copys the matrix and all internal data such that the old and the new matrix do not share any data.

        Returns
        -------
        :obj:`kron_matrix`
            copy of this matrix
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        blocks = np.empty(shape=self.bshape, dtype='object')
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                blocks[i,j] = self._blocks[i,j].copy()
        return self.__class__(blocks)
    
    def diagonal(self):
        """Returns the diagonal of the matrix.

        Returns
        -------
        :obj:`block_matrix`
        
        
        .. todo::
            Erweitern auf allgemeine block matrizen
            

        .. codeauthor:: Moritz Feuerle, 2022
        """
        if self.sbshape[0] != self.sbshape[1]:
            raise NotImplementedError("Right now, only block matricies with symmetric block sizes supported.")
        return self.__class__([[ self._blocks[i,i].diagonal() if iskronrelated(self._blocks[i,i]) else self._blocks[i,i].diagonal().reshape(1,-1) for i in range(self.bshape[0])]])
    
    def eliminate_zeros(self):
        r"""Eliminates zero entrys in the individual sub-blocks.
        
        Warning
        ----------
            This operation could not only affect this matrix, but also other 
            matricies that share the same internal matricies as in ``blocks``. 
            Nevertheless, this should not leed to any critical side effects, since only zeros are removed.
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                if hasattr(self._blocks[i,j], 'eliminate_zeros'):
                    self._blocks[i,j].eliminate_zeros()
            
            
    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)
        bs = 'bshape=' + str(self.bshape)

        return '<%dx%d %s with %s, %s>' % (M, N, self.__class__.__name__, bs, dt)


class kronblock_matrix(block_matrix):
    r"""Block matrix of Kronecker product matricies

    .. math::
        A = \begin{pmatrix} \texttt{blocks[0][0]} & \dots & \texttt{blocks[0][N-1]} \\
            \vdots & \ddots & \vdots \\
            \texttt{blocks[M-1][0]} & \dots & \texttt{blocks[M-1][N-1]}
            \end{pmatrix}.
    
    where each block is a :obj:`kron_matrix` or :obj:`kronsum_matrix` with compatible shapes.
    
    This is basicly the same as :obj:`block_matrix`, but it is ensured that all blocks are 
    Kronecker products with not only compatible shapes but also compatible :attr:`kshape <kron_matrix.kshape>`.

    The blocks are internally stored as is, i.e. the block matrix is not 
    assembled in memory. This is usefull for Kronecker shaped 
    matricies, since the Kronecker structure of each subblock would be lost by
    assebeling the whole matrix, resulting in a much larger memory consumption.        
        
    .. todo:: 
        self.dtype verwenden wenn matrix assambled wird oder neue arrays angelegt werden usw. 
    .. todo::
        sollten blöcke von blöcken beim init auch integriert werden in eine block struktur?
    """
    
    kdim : int
    r"""Number of concatinated Kronecker products (this is at least 2).
    
    Example: :math:`A \otimes B \otimes C` has ``kdim=3``.
    """
    
    bshape : tuple
    """Block shape ``bshape=(M,N)``"""
    
    sbshape : tuple
    """Sub-block :attr:`shape` ``blocks[i][j].shape = (sbshape[0][i], sbshape[1][j])``"""
    
    sbkshape : tuple
    """Sub-block :attr:`kshape <kron_matrix.kshape>` ``blocks[i][j].kshape = (sbkshape[0][i], sbkshape[1][j])``"""
    
    def __init__(self, blocks, copy=False):
        """
        Parameters
        ----------
        blocks : array_like
            Grid of Kronecker product matrices with compatible shapes and kshapes, has to be 2D. An entry of ``None`` implies an all-zero sub-block.
        copy
            If ``True``, the matricies in ``blocks`` will be copied.
            
            
        .. codeauthor:: Moritz Feuerle, 2022
        """
        
        blocks, dtype, sbshape, sbkshape, kdim = self._assemble_block_shape(blocks,copy)
        
        self._blocks  = blocks
        self._offsets = [np.append(0, np.cumsum(sbshape[i])) for i in range(2)]
        
        self.dtype = dtype
        self.shape = tuple(self._offsets[i][-1] for i in range(2))
        self.bshape = blocks.shape
        self.sbshape = sbshape
        self.sbkshape = sbkshape
        self.kdim = kdim
        
        
    def _assemble_block_shape(self,blocks,copy):      
        M = len(blocks)
        try:
            N = len(blocks[0])
        except TypeError:
            raise ValueError('blocks must be 2D')
            
        brow_lengths = np.zeros(M, dtype=np.int64)
        bcol_lengths = np.zeros(N, dtype=np.int64)
        
        kdim = 0
        brow_kshape = np.zeros(M, dtype=tuple)
        bcol_kshape = np.zeros(N, dtype=tuple)
        
        dtype = []
        
        for i in range(M):
            if len(blocks[i]) != N:
                raise ValueError(f"inconsisten block shape, expectet blocks.shape={(M,N)}, but len(blocks[{i}]) = {len(blocks[i])}")
            for j in range(N):
                if blocks[i][j] is not None:
                    A = blocks[i][j]
                    if A.ndim != 2:
                        raise ValueError("Only 2 dimensional matricies supported")
                    if not iskronshaped(A):
                        raise ValueError("Only kronecker shaped matricies allowed")
                    
                    dtype.append(A.dtype)
                    
                    if brow_lengths[i] == 0:
                        brow_lengths[i] = A.shape[0]
                    elif brow_lengths[i] != A.shape[0]:
                        msg = (f'blocks[{i},:] has incompatible row dimensions. '
                            f'Got blocks[{i},{j}].shape[0] == {A.shape[0]}, '
                            f'expected {brow_lengths[i]}.')
                        raise ValueError(msg)

                    if bcol_lengths[j] == 0:
                        bcol_lengths[j] = A.shape[1]
                    elif bcol_lengths[j] != A.shape[1]:
                        msg = (f'blocks[:,{j}] has incompatible column '
                            f'dimensions. '
                            f'Got blocks[{i},{j}].shape[1] == {A.shape[1]}, '
                            f'expected {bcol_lengths[j]}.')
                        raise ValueError(msg)
                    
                    if kdim == 0:
                        kdim = A.kdim
                    elif kdim != A.kdim:
                        msg = (f'blocks[{i},{j}] has incompatible kdim. '
                            f'Got blocks[{i},{j}].kdim == {A.kdim}, '
                            f'expected {kdim}.')
                        raise ValueError(msg)
                    
                    if brow_kshape[i] == 0:
                        brow_kshape[i] = A.kshape[0]
                    elif brow_kshape[i] != A.kshape[0]:
                        msg = (f'blocks[{i},:] has incompatible kronecker shape. '
                            f'Got blocks[{i},{j}].kshape[0] == {A.kshape[0]}, '
                            f'expected {brow_kshape[i]}.')
                        raise ValueError(msg)
                    
                    if bcol_kshape[j] == 0:
                        bcol_kshape[j] = A.kshape[1]
                    elif bcol_kshape[j] != A.kshape[1]:
                        msg = (f'blocks[:,{j}] has incompatible kronecker shape. '
                            f'Got blocks[{i},{j}].kshape[1] == {A.kshape[1]}, '
                            f'expected {bcol_kshape[j]}.')
                        raise ValueError(msg)
                    
        sbshape = tuple(brow_lengths), tuple(bcol_lengths)
        sbkshape = tuple(brow_kshape), tuple(bcol_kshape)
        dtype = np.dtype(np.result_type(*dtype))
        blocks_new = np.full((M,N), fill_value=None, dtype='object')
        for i in range(M):
            for j in range(N):
                if blocks[i][j] is None:
                    from ._construct import zeros
                    blocks_new[i,j] = zeros((sbkshape[0][i],sbkshape[1][j]), dtype=dtype)
                else:
                    blocks_new[i,j] = blocks[i][j].copy() if copy else blocks[i][j]
        
        return blocks_new, dtype, sbshape, sbkshape, kdim  
    
    
    ####################################################
    # access kronecker layers
    ####################################################
    def getklayer(self, k):
        """Returnes the matricies of the k-th Kronecker layer of each sub-block.
        
        Parameters
        ----------
        k
            Kronecker layer (``0 <= k <`` :attr:`kdim`)
        
        Returns
        ----------
        :obj:`block_matrix`
            ``out._blocks[i,j] = blocks[i,j].getklayer(k)``
        
        
        See Also
        --------
        :meth:`kron_matrix.getklayer`
        :meth:`kronsum_matrix.getklayer`
        
        
        .. codeauthor:: Moritz Feuerle, 2022
        """
        assert k in range(self.kdim)
        out = np.empty(shape=self.bshape, dtype='object')
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                out[i,j] = self._blocks[i,j].getklayer(k)
        return block_matrix(out)
          
    ####################################################
    # other
    ####################################################
    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)
        kd = 'kdim=' + str(self.kdim)
        bs = 'bshape=' + str(self.bshape)

        return '<%dx%d %s with %s, %s, %s>' % (M, N, self.__class__.__name__,kd, bs, dt) 


def iskron(x):
    """Is ``x`` a :obj:`kron_matrix`?
    
    Parameters
    ----------
    x
        object to check for being a Kronecker product matrix
        
    Returns
    -------
    bool
        ``True`` if ``x`` is a Kronecker product matrix, ``False`` otherwise
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return isinstance(x,kron_matrix)

def iskronsum(x):
    """Is ``x`` a :obj:`kronsum_matrix`?
    
    Parameters
    ----------
    x
        object to check for being a sum of Kronecker product matricies
        
    Returns
    -------
    bool
        ``True`` if ``x`` is a sum of Kronecker product matricies, ``False`` otherwise
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return isinstance(x,kronsum_matrix)

def iskronblock(x):
    """Is ``x`` a :obj:`kronblock_matrix`?
    
    Parameters
    ----------
    x
        object to check for being a Kronecker product block matrix
        
    Returns
    -------
    bool
        ``True`` if ``x`` is a Kronecker product block matricies, ``False`` otherwise
        
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return isinstance(x,kronblock_matrix)
                
def isblock(x):
    """Is ``x`` a :obj:`block_matrix` or :obj:`kronblock_matrix`?
    
    Parameters
    ----------
    x
        object to check for being a block matrix
        
    Returns
    -------
    bool
        ``True`` if ``x`` is a block matrix, ``False`` otherwise
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return isinstance(x,block_matrix)
# Moritz Feuerle, May 2022

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import issparse
from scipy.sparse._sputils import isscalarlike

from kron.utils import tosparray, iscollection
from ._interfaces import iskronrelated
from ._kron import kron_matrix, kronsum_matrix, kronblock_matrix, block_matrix, iskron, iskronsum, isblock


__all__ = ['kron', 'blockedkron', 'block', 'vstack', 'hstack', 'diag', 'eye', 'zeros', 'ones']


def kron(*A,copy=False):
    r"""Multi-layer Kronecker product (or Tensor product) structured matrix
    
    .. math::
        \texttt{A} := \texttt{A[0]} \otimes \texttt{A[1]} \otimes \dots \otimes \texttt{A[kdim-1]}.
        
    Can be called with a list of matricies or individual matricies
        
    - ``kron(A1,A2,...)``
    - ``kron(A1,A2,... , copy=copy)``
    - ``kron(A)`` with ``A = [A1,A2,...]``
    - ``kron(A, copy=copy)`` with ``A = [A1,A2,...]``.
    
    If ``A1,A2,...`` are :obj:`utils.MatrixCollection` with same length, the Kronecker product is taken
    elementwise and the sum is returned, i.e. ``kron(A1[0],A2[0],...) + kron(A1[1],A2[1],...) + ...``.
    
       
    Parameters
    ----------
    A
        List of matricies ``A[0],...,A[kdim-1]``.
        Each matrix ``A[i]`` could be either a :obj:`numpy.ndarray`, 
        :obj:`scipy.sparse.sparray`, another :obj:`kron_matrix` or a 
        :obj:`utils.MatrixCollection`.
    copy
        If ``True``, the matricies ``A[0],...,A[kdim-1]`` will be copied.
        
    Returns
    -------
    :obj:`kron_matrix` or :obj:`kronsum_matrix`
        Kronecker product of the given matricies
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    if len(A) == 1:
        A = A[0]
        
    if any(iscollection(Ai) for Ai in A):
        L = 0
        for i in range(len(A)):
            if iscollection(A[i]):
                if L == 0:
                    L = len(A[i])
                elif L != len(A[i]):
                    raise ValueError(f"Collections have to have te same length, collection {i} has len={len(A[i])}, expected len={L}.")
                
        return sum(kron([Ai[l] if iscollection(Ai) else Ai for Ai in A],copy=copy) for l in range(L))
    
    return kron_matrix(A,copy=copy)


def blockedkron(*A, copy=False):
    r"""Blocked multi-layer Kronecker product (or Tensor product) of two or more :obj:`block_matrix`,
    where the Kronecker product is applied blockwise, i.e. for input matricies :math:`A_1,A_2,\dots` with
    
    .. math::
        A_i = \begin{pmatrix} \texttt{blocks}_i\texttt{[0][0]} & \dots & \texttt{blocks}_i\texttt{[0][N-1]} \\
            \vdots & \ddots & \vdots \\
            \texttt{blocks}_i\texttt{[M-1][0]} & \dots & \texttt{blocks}_i\texttt{[M-1][N-1]}
            \end{pmatrix},
    
    the output reads
    
    .. math::
        A_\text{out} = \begin{pmatrix} \texttt{blocks}_\text{out}\texttt{[0][0]} & \dots & \texttt{blocks}_\text{out}\texttt{[0][N-1]} \\
            \vdots & \ddots & \vdots \\
            \texttt{blocks}_\text{out}\texttt{[M-1][0]} & \dots & \texttt{blocks}_\text{out}\texttt{[M-1][N-1]}
            \end{pmatrix},
    
    with
    
    .. math::
        \texttt{blocks}_\text{out}\texttt{[k][l]} := \texttt{blocks}_1\texttt{[k][l]} \otimes \texttt{blocks}_2\texttt{[k][l]} \otimes \dots .
        
    Can be called with a list of matricies or individual matricies
    
    - ``blockedkron(A1,A2,...)``
    - ``blockedkron(A1,A2,... , copy=copy)``
    - ``blockedkron(A)`` with ``A = [A1,A2,...]``
    - ``blockedkron(A, copy=copy)`` with ``A = [A1,A2,...]``.
    
    Parameters
    ----------
    Parameters
    ----------
    A
        List of :obj:`block_matrix`
    copy
        If ``True``, the matricies in ``A`` will be copied.
        
    Returns
    -------
    :obj:`kronblock_matrix`
        Blockwise Kronecker product of the given matricies
        
        
    .. codeauthor:: Moritz Feuerle, 2022
    """
    if len(A)==1:
        A = A[0]
        
    for i in range(len(A)):
        if not isblock(A[i]):
            A[i] = block(A[i])
            
    assert all(isblock(Ai) for Ai in A)
    M,N = A[0].bshape
    assert all((M,N) == Ai.bshape for Ai in A)
    if copy:
        A = [Ai.copy() for Ai in A]
    out = np.empty((M,N), dtype='object')
    for i in range(M):
        for j in range(N):
            out[i,j] = kron([Ai._blocks[i,j] for Ai in A])
    return kronblock_matrix(out)
    

def block(blocks, copy=False, sbshape=None):
    """Build a block matrix from subblocks. If possible, a :obj:`kronblock_matrix` is returned,
    otherwise :obj:`block_matrix`.
    
    Can be called in two seperate ways:
    
    1. With a already blocked matrix in ``blocks`` and ``sbshape=None``. In this case the given blocks will be used to build the block matrix.
    2. With a normal matrix in ``blocks`` and ``sbshape`` specifing the shape of the sub-blocks. In this case the given matrix will be partitioned with the given shape

    Parameters
    ----------
    blocks : array_like
        Grid of matrices with compatible shapes. An entry of ``None`` implies an all-zero matrix.
    copy
        If ``True``, the matricies in ``blocks`` will be copied.
    sbshape
        If this is given, blocks should be a single matrix, which will be part according to sbshape
            
    Returns
    -------
    :obj:`kronblock_matrix` or :obj:`block_matrix`
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    
    if sbshape is not None:
        A = blocks
        
        sbshape = list(sbshape)
        if sbshape[0] == 1:
            sbshape[0] = [1]
        elif sbshape[1] == 1:
            sbshape[1] = [1]
            
        if A.ndim == 1:
            A = A.reshape((1,-1)) if len(sbshape[0])==1 else A.reshape((-1,1))
            
        offsets = np.append(0,np.cumsum(sbshape[0])), np.append(0,np.cumsum(sbshape[1]))
        assert A.shape == (offsets[0][-1],offsets[1][-1])
        
        M = len(sbshape[0])
        N = len(sbshape[1])
        blocks = np.empty((M,N), dtype='object')
        for i in range(M):
            i_idx = slice(offsets[0][i], offsets[0][i+1])
            for j in range(N):
                j_idx = slice(offsets[1][j], offsets[1][j+1])
                blocks[i,j] = A[i_idx,:][:,j_idx]
    
    try:
        return kronblock_matrix(blocks,copy=copy)
    except:
        return block_matrix(blocks,copy=copy)

def vstack(blocks, copy=False):
    """Stack matrices vertically (row wise). If possible, a :obj:`kronblock_matrix` is returned,
    otherwise :obj:`block_matrix`.

    Parameters
    ----------
    blocks : array_like
        sequence of matrices with compatible shapes
    copy
        If ``True``, the matricies in ``blocks`` will be copied.
            
    Returns
    -------
    :obj:`kronblock_matrix` or :obj:`block_matrix`
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return block([[b] for b in blocks],copy=copy)

def hstack(blocks, copy=False):
    """Stack matrices horizontally (column wise). If possible, a :obj:`kronblock_matrix` is returned,
    otherwise :obj:`block_matrix`.

    Parameters
    ----------
    blocks : array_like
        sequence of matrices with compatible shapes
    copy
        If ``True``, the matricies in ``blocks`` will be copied.
            
    Returns
    -------
    :obj:`kronblock_matrix` or :obj:`block_matrix`
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    return block([blocks],copy=copy)

def diag(diagonal):
    """Returns a diagonal matrix from de given diagonal.
    
    This is basicly :func:`scipy.sparse.diags`, but can also construct matricies from
    vectors of the type :obj:`kron_matrix`, :obj:`kronsum_matrix` and :obj:`block_matrix`.

    Parameters
    ----------
    diagonal : array_like
        vector to be placed on the diagonal
    
    Returns
    -------
    :obj:`scipy.sparse.sparray`, :obj:`kron_matrix`, :obj:`kronsum_matrix` and :obj:`block_matrix`
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    
    if iskronrelated(diagonal):
        if diagonal.shape[0] != 1 and diagonal.shape[1] != 1:
                raise ValueError("not a vector")
        if iskron(diagonal):
            return kron_matrix([diag(A) for A in diagonal._data])
        if iskronsum(diagonal):
            return kronsum_matrix([diag(A) for A in diagonal._data])
        if isblock(diagonal):
            bshape = np.max(diagonal.bshape)
            blocks = np.full(shape=(bshape,bshape), fill_value=None, dtype='object')  
            for i in range(bshape):
                if diagonal.shape[0] == 1:
                    blocks[i,i] = diag(diagonal._blocks[0][i])
                else:
                    blocks[i,i] = diag(diagonal._blocks[i][0])
            return block_matrix(blocks)
    elif issparse(diagonal):
        diagonal = diagonal.toarray()
        
    if diagonal.ndim == 2 and (diagonal.shape[0] == 1 or diagonal.shape[1] == 1):
        diagonal = diagonal.reshape(-1)
    return tosparray(sparse.diags(diagonal))

    
def eye(m, n=None, k=0, dtype=float, format=None):
    """Sparse or Kronecker matrix with ones on diagonal.

    If m (and n) are scalar, this method is identical with :meth:`scipy.sparse.eye`
    and returns a matrix with ``shape = (m,n)``.
    
    If m (and n) are arrays a :obj:`kron_matrix` is returned with :attr:`Kronecker shape <kron_matrix.kshape>`
    ``kshape = (m,n)``. In this case, ``format`` could be either a string or a list of formats with different
    formats for for each submatrix of the Kronecker product.

    Parameters
    ----------
    m : int or array_like
        Number of rows in the matrix or in each submatrix of a Kronecker shaped matrix.
    n : int or array_like, optional
        Number of columns or in each submatrix of a Kronecker shaped matrix. Default: m.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal). Values other than 0 are not
        supported for construction of a Kronecker shaped matrix.
    dtype : :obj:`numpy.dtype`, optional
        Data type of the matrix.
    format : str (or list of strings), optional
        Sparse format, e.g., format=”csr”, etc.

    Returns
    -------
    :obj:`kron_matrix` or :obj:`scipy.sparse.sparray`
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    if n is None: n = m
    
    if isscalarlike(m) and isscalarlike(n):
        return sparse.eye(m=m,n=n,k=k,dtype=dtype,format=format)
    
    assert len(m) == len(n)
    if k != 0:
        raise NotImplementedError("off-diagonals not implemented for kronecker matricies")
    if isinstance(format,str) or format is None:
        format = len(m) * [format]
    
    A = []
    for i in range(len(m)):
        A.append(sparse.eye(m=m[i], n=n[i], dtype=dtype, format=format[i]))
    return kron(A,copy=False)
    
    
def zeros(shape, dtype=float, format='coo'):
    """All zeros matrix. 
    
    ``shape`` can either be a normal matrix shape (tuple)
    or a kshape (tupel of tuples) to construct a all zeros
    :obj:`scipy.sparse.sparray` or a all zeros :obj:`kron_matrix`.
    ``format`` could be either a string or a list of formats with different
    formats for for each submatrix of the :obj:`kron_matrix`.

    Parameters
    ----------
    shape : tuple or tuple of tuples
        shape or :attr:`kron_matrix.kshape` of the matrix
    dtype : :obj:`numpy.dtype`, optional
        Data type of the matrix.
    format : str (or list of strings), optional
        Sparse format, e.g., format=”csr”, etc.

    Returns
    -------
    :obj:`kron_matrix` or :obj:`scipy.sparse.sparray`
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    
    if isscalarlike(shape[0]):
        return sparse.coo_array(shape,dtype=dtype).asformat(format)
    
    if isinstance(format,str):
        format = len(shape[0]) * [format]
        
    if len(shape) == 1:
        A = [sparse.coo_array((shape[0][i],),dtype=dtype).asformat(format[i]) for i in range(len(shape[0]))]
    else:
        assert len(shape[0]) == len(shape[1])
        A = [sparse.coo_array((shape[0][i],shape[1][i]),dtype=dtype).asformat(format[i]) for i in range(len(shape[0]))]
    return kron_matrix(A)


def ones(shape, *args, **kwargs):
    """All ones matrix. 
    
    ``shape`` can either be a normal matrix shape (tuple)
    or a kshape (tupel of tuples) to construct a all ones
    :obj:`scipy.sparse.sparray` or a all ones :obj:`kron_matrix`.

    Parameters
    ----------
    shape : tuple or tuple of tuples
        shape or :attr:`kron_matrix.kshape` of the matrix
    others
        all other possible arguments of :meth:`numpy.ones`

    Returns
    -------
    :obj:`kron_matrix` or :obj:`numpy.ndarray`
    
    
    .. codeauthor:: Moritz Feuerle, 2022
    """
    
    if isscalarlike(shape[0]):
        return np.ones(shape,*args,**kwargs)

    if len(shape) == 1:
        A = [np.ones((shape[0][i],),*args,**kwargs) for i in range(len(shape[0]))]
    else:
        assert len(shape[0]) == len(shape[1])
        A = [np.ones((shape[0][i],shape[1][i]),*args,**kwargs) for i in range(len(shape[0]))]
    return kron_matrix(A)
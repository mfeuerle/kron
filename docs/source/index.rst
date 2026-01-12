.. Kron documentation master file, created by
   sphinx-quickstart on Wed Mar 30 00:11:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Kronecker product Python module
=====================================================


This linear algebra module is for construction of matricies with 
Kronecker product (or Tensor product) structure
    
.. math::
    A := A_1 \otimes A_2 \otimes \dots \otimes A_{k}

where the Kronecker product :math:`A \otimes B \in \mathbb{R}^{np \times mq}` of two matricies 
:math:`A \in \mathbb{R}^{n \times m}` and :math:`B \in \mathbb{R}^{p \times q}` is defined as 
    
.. math::
    A \otimes B := \begin{pmatrix} a_{11}B & \dots & a_{1m}B \\
                    \vdots & \ddots & \vdots \\
                    a_{n1}B & \dots & a_{nm}B
                    \end{pmatrix}.

This module is desigend to only work with the subparts :math:`A_1,...,A_{k}`. Hence, the full matrix :math:`A`
will never be assembled during. If it is necessary to assemble the full matrix, this has to be done explicitly 
by the user.

As base matricies for :math:`A_1,...,A_{k}` are :obj:`scipy.sparse.sparray` and :obj:`numpy.ndarray`
supported.

All matrix classes in this module can be wraped as :obj:`scipy.sparse.linalg.LinearOperator` using
:obj:`scipy.sparse.linalg.aslinearoperator`, i.e. many methods from 
:mod:`scipy.sparse.linalg` can be used directly, e.g. :meth:`scipy.sparse.linalg.cg` or :meth:`scipy.sparse.linalg.gmres`.
Normally this is already done at the start of such a method, i.e. one does not 
need to call :obj:`scipy.sparse.linalg.aslinearoperator` manually.

It is recommended to use the new :obj:`scipy.sparse.sparray` instead of the deprecated
:obj:`scipy.sparse.spmatrix` (like :obj:`scipy.sparse.csr_array` instead of :obj:`scipy.sparse.csr_matrix`) if one uses sparse
matricies. Main differece: :obj:`scipy.sparse.sparray` support elementwise multiplication,
hence this module spports elementwise multiplication only for those. 
Either way, all sparse matricies returned by this module are :obj:`scipy.sparse.sparray`.
Beside of elementwise multiplication, everything else should work with :obj:`scipy.sparse.spmatrix` 
as well, but note that this framework is only test in combination with :obj:`scipy.sparse.sparray`.


All matricies implement the 

   - elementwise multiplication operator ``*``
   - matrix multiplication operator ``@``
   - addition operator ``+``
   - substraction and negation operator ``-``
   

All matricies passed this module will be taken by reference for efficency, i.e. do not modify 
any matrix after it was passed to this module since it will corrupt the data.
If you want to modify any matrix make sure that they are copyed befor doing so.

As an example, in the following code is not only A changed, but also AB and S corrupt:

>>> A = np.array([[1,2],[3,4]])
>>> B = np.array([[1,2],[3,4]])
>>> AB = kron.kron(A,B)
>>> AB.toarray()
array([[ 1,  2,  2,  4],
       [ 3,  4,  6,  8],
       [ 3,  6,  4,  8],
       [ 9, 12, 12, 16]])
>>> S = AB + AB
>>> S.toarray()
array([[ 2,  4,  4,  8],
       [ 6,  8, 12, 16],
       [ 6, 12,  8, 16],
       [18, 24, 24, 32]])
>>> A[0,0] = 0
>>> AB.toarray()
array([[ 0,  0,  2,  4],
       [ 0,  0,  6,  8],
       [ 3,  6,  4,  8],
       [ 9, 12, 12, 16]])
>>> S.toarray()
array([[ 0,  0,  4,  8],
       [ 0,  0, 12, 16],
       [ 6, 12,  8, 16],
       [18, 24, 24, 32]])

To prevent this use eiter

>>> AB = kron.kron(A,B, copy=True)

or manually copy A bevor changing its contents:

>>> A = A.copy()
>>> A[0,0] = 0



Since :mod:`scipy.sparse` does not resepect ``__array_priority__`` when importing this module
the :func:`kron.utils.patch_scipy_array_priority` method is automatically called. 


Kronecker product module
~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   modules/kron
   modules/kron.utils


:ref:`genindex`







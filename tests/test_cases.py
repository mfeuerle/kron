#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 2022

Execute via pytest (just run `pytest` on the console)

pytest searches for all files in the directory and subdirectory, matching
test_*.py or *_test.py and executes all functions with name test* or *test
as well as all object methods object.test* or object.*test from classes with
the name Test*

@author: Moritz Feuerle
"""

import kron
from kron.utils import tosparray
import itertools
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import uniform



############################################################################################
############################################################################################
#
#   MATRIX GENERATORS
#
############################################################################################
############################################################################################

##########################################
# General stuff
##########################################
    
class _MatrixGenerator_Interface():
    MAT_DIM   = [1, 5]      # dimension range for a random numpy or scipy matrix
    BLOCK_DIM = [1, 3]      # number of blocks for a random blocked matrix
    
    ### to be done ####
    def get_new_matrix(self,test_settings,squared=False):
        pass
    
    def _get_matrix_with_shape(self,matrixL,idxL,matrixR,idxR):
        pass
    
    def _get_vector_with_shape(self,matrix,idx):
        pass


    ### already done
    def rand_dim(self):
        return np.random.randint(self.MAT_DIM[0], self.MAT_DIM[1])
    
    def rand_blk_dim(self):
        return np.random.randint(self.BLOCK_DIM[0], self.BLOCK_DIM[1])

    def get_basic_matrix(self, shape0, shape1, format):
        if shape0 is None:
            shape0 = self.rand_dim()
        if shape1 is None:
            shape1 = self.rand_dim()    
        A = tosparray(sparse.random(shape0, shape1, density=0.5, data_rvs=uniform(loc=-5,scale=10).rvs))
        return A.asformat(format), A.toarray() 
    
    
    def get_matrix_with_shape(self,matrixL=None,idxL=None,matrixR=None,idxR=None):
        assert matrixL is not None or matrixR is not None
        return self._get_matrix_with_shape(matrixL,idxL,matrixR,idxR)
    
    def get_matrix_same_shape(self,matrix):
        return self.get_matrix_with_shape(matrixL=matrix,idxL=0,matrixR=matrix,idxR=1)
    
    
    def get_mul_matrix(self,matrix):
        return self.get_matrix_same_shape(matrix)
        
    def get_rmul_matrix(self,matrix):
        return self.get_matrix_same_shape(matrix)
    
    def get_mul_vector(self,matrix):
        for A,A_ref in self._get_vector_with_shape(matrix=matrix,idx=1):
            yield A.T, A_ref.T 
    
    def get_rmul_vector(self,matrix):
        return self._get_vector_with_shape(matrix=matrix,idx=0)
    
    def get_matmul_matrix(self,matrix):
        return self.get_matrix_with_shape(matrixL=matrix,idxL=1,matrixR=None,idxR=None)
            
    def get_rmatmul_matrix(self,matrix):
        return self.get_matrix_with_shape(matrixL=None,idxL=None,matrixR=matrix,idxR=0)
            
    def get_add_matrix(self,matrix):
        return self.get_matrix_same_shape(matrix)
            
    def get_radd_matrix(self,matrix):
        return self.get_matrix_same_shape(matrix)
            
    def get_sub_matrix(self,matrix):
        return self.get_matrix_same_shape(matrix)
            
    def get_rsub_matrix(self,matrix):
        return self.get_matrix_same_shape(matrix)
            
            
##########################################
# Normal matrix generators
##########################################
class _NormalMatrixGenerator_Interface(_MatrixGenerator_Interface):
    ### to be specified ###
    formats   = None
    
    def get_new_matrix(self,test_settings,squared=False):
        for format in self.formats:
            d1 = self.rand_dim()
            d2 = d1 if squared else self.rand_dim()
            yield self.get_basic_matrix(d1,d2, format)
    
    def _get_matrix_with_shape(self,matrixL,idxL,matrixR,idxR):
        if matrixL is None:
            shape0 = self.rand_dim()
        else:
            shape0 = matrixL.shape[idxL]
        if matrixR is None:
            shape1 = self.rand_dim()
        else:
            shape1 = matrixR.shape[idxR]
            
        for format in self.formats:
            yield self.get_basic_matrix(shape0,shape1, format)
            
    def _get_vector_with_shape(self,matrix,idx):
        if matrix is None:
            shape0 = self.rand_dim()
        else:
            shape0 = matrix.shape[idx]
                
        for format in self.formats:
            yield self.get_basic_matrix(shape0,1, format)
            
                  
class DenseMatrixGenerator(_NormalMatrixGenerator_Interface):
    formats = ['array']
class SparseMatrixGenerator(_NormalMatrixGenerator_Interface):
    formats = ['coo','csr','csc','bsr','lil']
class SparseDenseMatrixGenerator(_NormalMatrixGenerator_Interface):
    formats = ['array','csr']
    

##########################################
# kron_matrix generators
##########################################
class _KronMatrixGenerator_Interface(_MatrixGenerator_Interface):
    ### to be specified ###
    internal_formats = None
    
    ### actual implementation ##      
    def _get_matrix(self, kdim, kshape):
        formats = [self.internal_formats for d in range(kdim)]
        for format in list(itertools.product(*formats)):
            A,A_ref = self.get_basic_matrix(kshape[0][0],kshape[1][0], format[0])
            A = [A]
            for d in range(1,kdim):
                B,B_ref = self.get_basic_matrix(kshape[0][d],kshape[1][d], format[d])
                A = A + [B]
                A_ref = np.kron(A_ref,B_ref)
            yield kron.kron(A), A_ref
          
               
    def get_new_matrix(self,test_settings,squared=False):
        kdim = test_settings.kdim
        if squared:
            kshape = 2 * [[self.rand_dim() for i in range(kdim)]]
        else:
            kshape = [[None for i in range(kdim)],[None for i in range(kdim)]]
        return self._get_matrix(kdim, kshape)
    
    def _get_matrix_with_shape(self,matrixL,idxL,matrixR,idxR):
        kdim = None
        kshape = [None,None]
        if matrixL is not None:
            if kron.iskronshaped(matrixL):
                kdim = matrixL.kdim
                kshape[0] = matrixL.kshape[idxL]
            elif kron.isblock(matrixL):
                kdim = 2
                if matrixL.shape[idxL] == 1:
                    kshape[0] = [1,1]
                else:
                    tmp = matrixL.shape[idxL]-1
                    while matrixL.shape[idxL] % tmp != 0:
                        tmp -= 1
                    kshape[0] = [tmp, int(matrixL.shape[idxL] / tmp)]   
            else: raise NotImplementedError(f"Dont know how to handel {type(matrixL)}")
        if matrixR is not None:
            if kron.iskronshaped(matrixR):
                if kdim is not None: assert kdim == matrixR.kdim
                kdim = matrixR.kdim
                kshape[1] = matrixR.kshape[idxR]
            elif kron.isblock(matrixR):
                kdim = 2
                if matrixR.shape[idxR] == 1:
                    kshape[1] = [1,1]
                else:
                    tmp = matrixR.shape[idxR]-1
                    while matrixR.shape[idxR] % tmp != 0:
                        tmp -= 1
                    kshape[1] = [tmp, int(matrixR.shape[idxR] / tmp)]
            else: raise NotImplementedError(f"Dont know how to handel {type(matrixR)}")
        if kshape[0] is None: kshape[0] = [None for i in range(kdim)]
        if kshape[1] is None: kshape[1] = [None for i in range(kdim)]
        return self._get_matrix(kdim, kshape)      
    
    def _get_vector_with_shape(self,matrix,idx):
        kdim = None
        kshape0 = None
        if matrix is not None:
            if kron.iskronshaped(matrix):
                kdim = matrix.kdim
                kshape0 = matrix.kshape[idx]
            elif kron.isblock(matrix):
                kdim = 2
                if matrix.shape[idx] == 1:
                    kshape0 = [1,1]
                else:
                    tmp = matrix.shape[idx]-1
                    while matrix.shape[idx] % tmp != 0:
                        tmp -= 1
                    kshape0 = [tmp, int(matrix.shape[idx] / tmp)]   
            else: raise NotImplementedError(f"Dont know how to handel {type(matrix)}")
        if kshape0 is None: kshape0 = [None for i in range(kdim)]
        kshape1 = [1 for i in range(kdim)]
        return self._get_matrix(kdim, (kshape0,kshape1))
            
class BasicKronMatrixGenerator(_KronMatrixGenerator_Interface):
    internal_formats = ['csr','array']
class BasicKronMatrixGenerator_Large(_KronMatrixGenerator_Interface):
    internal_formats = ['coo','csr','csc','array','bsr','lil']
    
    
##########################################
# kronsum_matrix generators
########################################## 
class _KronSumMatrixGenerator_Interface(_MatrixGenerator_Interface):
    ### to be specified ###
    matrix_generator_init = None
    matrix_generator_add  = None
    
    ### actual implementation ##
    def get_new_matrix(self,test_settings,squared=False):
        for A,A_ref in self.matrix_generator_init.get_new_matrix(test_settings,squared=squared):
            for B,B_ref in self.matrix_generator_add.get_add_matrix(A):
                A += B
                A_ref += B_ref
            yield A, A_ref
            
    def _get_matrix_with_shape(self,matrixL,idxL,matrixR,idxR):
        for A,A_ref in self.matrix_generator_init.get_matrix_with_shape(matrixL,idxL,matrixR,idxR):
            for B,B_ref in self.matrix_generator_add.get_add_matrix(A):
                A += B
                A_ref += B_ref
            yield A, A_ref
    
    def _get_vector_with_shape(self,matrix,idx):
        for A,A_ref in self.matrix_generator_init._get_vector_with_shape(matrix,idx):
            for B,B_ref in self.matrix_generator_add.get_add_matrix(A):
                A += B
                A_ref += B_ref
            yield A, A_ref
            
class BasicKronSumMatrixGenerator(_KronSumMatrixGenerator_Interface):
    matrix_generator_init = BasicKronMatrixGenerator()
    matrix_generator_add  = BasicKronMatrixGenerator()
    
    

##########################################
# block_matrix generators
########################################## 
class _BlockMatrixGenerator_Interface(_MatrixGenerator_Interface):
    ### to be specified ###
    matrix_generator = None
    with_none = None
    kron_block = None
    
    ### actual implementation ##
    def _get_matrix(self, bshape, referenzL, referenzR):
        block_generators = np.empty(shape=bshape, dtype='object')
        for i in range(bshape[0]):
            for j in range(bshape[1]):
                block_generators[i,j] = self.matrix_generator.get_matrix_with_shape(referenzL[i],0,referenzR[j],1)
    
        for data in zip(*block_generators.flat):
            blocks = np.empty(shape=bshape, dtype='object')
            blocks_ref = np.empty(shape=bshape, dtype='object')
            for i in range(bshape[0]):
                for j in range(bshape[1]):
                    blocks[i,j] = data[i*bshape[1]+j][0]
                    blocks_ref[i,j] = data[i*bshape[1]+j][1]
            if self.with_none:
                i = np.random.randint(0,bshape[0])    
                j = np.random.randint(0,bshape[1])
                blocks[i,j] = None
                blocks_ref[i,j] = np.zeros(shape=blocks_ref[i,j].shape)
            if self.kron_block:
                yield kron.kronblock_matrix(blocks.tolist()), np.block(blocks_ref.tolist())
            else:
                yield kron.block_matrix(blocks.tolist()), np.block(blocks_ref.tolist())
            
    def get_new_matrix(self,test_settings,squared=False):
        if squared:
            bshape = 2 * [self.rand_blk_dim()]
            referenzL = [next(self.matrix_generator.get_new_matrix(test_settings,squared=True))[0] for i in range(bshape[0])]
            referenzR = referenzL
            
        else:
            bshape = [self.rand_blk_dim(), self.rand_blk_dim()]
            referenzL = [next(self.matrix_generator.get_new_matrix(test_settings))[0] for i in range(bshape[0])]
            referenzR = [next(self.matrix_generator.get_new_matrix(test_settings))[0] for i in range(bshape[1])]
        return self._get_matrix(bshape,referenzL,referenzR)
    
    def _get_matrix_with_shape(self,matrixL,idxL,matrixR,idxR):
        bshape = [None, None]
        if matrixL is not None:
            if kron.isblock(matrixL):
                bshape[0] = matrixL.bshape[idxL]
            else: raise NotImplementedError(f"Dont know how to handel {type(matrixL)}")
        if matrixR is not None:
            if kron.isblock(matrixR):
                bshape[1] = matrixR.bshape[idxR]
            else: raise NotImplementedError(f"Dont know how to handel {type(matrixR)}")
        
        if bshape[0] is None:
            bshape[0] = self.rand_blk_dim()
            matrixL = [None for i in range(bshape[0])]
        else:
            matrixL = [matrixL._blocks[i][0] if idxL==0 else matrixL._blocks[0][i] for i in range(bshape[0])]
        if bshape[1] is None:
            bshape[1] = self.rand_blk_dim()
            matrixR = [None for i in range(bshape[1])]
        else:
            matrixR = [matrixR._blocks[i][0] if idxR==0 else matrixR._blocks[0][i] for i in range(bshape[1])]
            
        referenzL = [next(self.matrix_generator.get_matrix_with_shape(matrixL[i],idxL,matrixR[0],idxR))[0] for i in range(bshape[0])]
        referenzR = [next(self.matrix_generator.get_matrix_with_shape(matrixL[0],idxL,matrixR[i],idxR))[0] for i in range(bshape[1])]
        return self._get_matrix(bshape,referenzL,referenzR)
    
    def _get_vector_with_shape(self,matrix,idx):
        bshape0 = None
        if matrix is not None:
            if kron.isblock(matrix):
                bshape0 = matrix.bshape[idx]
            else: raise NotImplementedError(f"Dont know how to handel {type(matrix)}")
    
        if bshape0 is None:
            bshape0 = self.rand_blk_dim()
            matrix = [None for i in range(bshape0)]
        else:
            matrix = [matrix._blocks[i][0] if idx==0 else matrix._blocks[0][i] for i in range(bshape0)]
            
        referenz = [next(self.matrix_generator._get_vector_with_shape(matrix[i],idx))[0] for i in range(bshape0)]
        return self._get_matrix((bshape0,1),referenz,[referenz[0]])
            
class BasicBlockMatrixGenerator(_BlockMatrixGenerator_Interface):
    matrix_generator = SparseDenseMatrixGenerator()
    with_None = True
    kron_block = False
    
class KronBlockMatrixGenerator(_BlockMatrixGenerator_Interface):
    matrix_generator = BasicKronMatrixGenerator()
    with_None = True
    kron_block = True


############################################################################################
############################################################################################
#
#   GENERAL TEST CASE SETTINGS
#
############################################################################################
############################################################################################  

class _Test_Settings():
    TOL_FACTOR = 5
    ABS_TOL    = 1e-13
    
    kdim = None
    bshape = None
     
    def get_tol(self,rel_tol):
        return max(self.TOL_FACTOR*rel_tol, self.ABS_TOL)
        
############################################################
#
#   TEST IF TWO MATRIX TYPES WORK TOGETHER
#
#############################################################

class _InterplayMatrixTester(_Test_Settings):
    """Base class for all kron_matrix test runs. Classes only need to
    specify the KronMatrixGenerator, which generates test matricies.
    """
    matrix_generator_1 = None
    matrix_generator_2 = None
    
    mul_supported = True
    matmul_supported = True
    add_supported = True
    sub_supported = True
    
    def generator_1(self,):
        return self.matrix_generator_1
    def generator_2(self,):
        return self.matrix_generator_2
    

    def test_mul(self):
        self._test_mul()
        self._test_rmul()
          
    def _test_mul(self):
        for A,A_ref in self.generator_1().get_new_matrix(self):
            ###################################
            # multiply with other matrix
            ###################################
            for B,B_ref in self.generator_2().get_mul_matrix(A):
                AB     = A     * B
                AB_ref = A_ref * B_ref
                try:
                    diff   = AB.toarray() - AB_ref
                except AttributeError:
                    diff   = AB - AB_ref
                tol    = np.finfo(diff.dtype).eps * np.linalg.norm(AB_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
            ###################################
            # multiply with other vector
            ###################################    
            for b,b_ref in self.generator_2().get_mul_vector(A):
                AB     = A     * b
                AB_ref = A_ref * b_ref
                try:
                    diff   = AB.toarray() - AB_ref
                except AttributeError:
                    diff   = AB - AB_ref
                tol    = np.finfo(diff.dtype).eps * np.linalg.norm(AB_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)

    def _test_rmul(self):
        for A,A_ref in self.generator_1().get_new_matrix(self):
            ###################################
            # multiply with other matrix
            ###################################
            for B,B_ref in self.generator_2().get_rmul_matrix(A):
                AB     = B     * A
                AB_ref = B_ref * A_ref
                diff   = AB.toarray() - AB_ref
                tol    = np.finfo(diff.dtype).eps * np.linalg.norm(AB_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
            ###################################
            # multiply with other vector
            ###################################    
            for b,b_ref in self.generator_2().get_rmul_vector(A):
                AB     = b     * A
                AB_ref = b_ref * A_ref
                diff   = AB.toarray() - AB_ref
                tol    = np.finfo(diff.dtype).eps * np.linalg.norm(AB_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
    

    def test_matmul(self):
        self._test_matmul()
        self._test_rmatmul()
              
    def _test_matmul(self):
        for A,A_ref in self.generator_1().get_new_matrix(self):
            ###################################
            # multiply with other matrix or vector
            ###################################
            for B,B_ref in self.generator_2().get_matmul_matrix(A):
                AB     = A     @ B
                AB_ref = A_ref @ B_ref
                try:
                    diff   = AB.toarray() - AB_ref
                except AttributeError:
                    diff   = AB - AB_ref
                tol    = np.finfo(diff.dtype).eps * np.linalg.norm(AB_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)

    def _test_rmatmul(self):
        for A,A_ref in self.generator_1().get_new_matrix(self):
            ###################################
            # multiply with other matrix or vector
            ###################################
            for B,B_ref in self.generator_2().get_rmatmul_matrix(A):
                AB     = B     @ A
                AB_ref = B_ref @ A_ref
                try:
                    diff   = AB.toarray() - AB_ref
                except AttributeError:
                    diff   = AB - AB_ref
                tol    = np.finfo(diff.dtype).eps * np.linalg.norm(AB_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)            
    

    def test_add(self):
        self._test_add()
        self._test_radd()
         
    def _test_add(self):
        ###################################
        # add other matrix
        ###################################
        for A,A_ref in self.generator_1().get_new_matrix(self):
            for B,B_ref in self.generator_2().get_add_matrix(A):
                A     = A     + B
                A_ref = A_ref + B_ref
                diff  = A.toarray() - A_ref
                tol   = np.finfo(diff.dtype).eps * np.linalg.norm(A_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
          
    def _test_radd(self):
        ###################################
        # add other matrix
        ###################################
        for A,A_ref in self.generator_1().get_new_matrix(self):
            for B,B_ref in self.generator_2().get_radd_matrix(A):
                A     = B     + A
                A_ref = B_ref + A_ref
                diff  = A.toarray() - A_ref
                tol   = np.finfo(diff.dtype).eps * np.linalg.norm(A_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
    

    def test_sub(self):
        self._test_sub()
        self._test_rsub()
  
    def _test_sub(self):
        ###################################
        # sub other matrix
        ###################################
        for A,A_ref in self.generator_1().get_new_matrix(self):
            for B,B_ref in self.generator_2().get_sub_matrix(A):
                A     = A     - B
                A_ref = A_ref - B_ref
                diff  = A.toarray() - A_ref
                tol   = np.finfo(diff.dtype).eps * np.linalg.norm(A_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
          
    def _test_rsub(self):
        ###################################
        # sub other matrix
        ###################################
        for A,A_ref in self.generator_1().get_new_matrix(self):
            for B,B_ref in self.generator_2().get_rsub_matrix(A):
                A     = B     - A
                A_ref = B_ref - A_ref
                diff  = A.toarray() - A_ref
                tol   = np.finfo(diff.dtype).eps * np.linalg.norm(A_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
                


############################################################
#
#   TEST A SINGLE MATRIX TYPE IN ITSELF
#
#############################################################

class _IsolatedMatrixTester(_InterplayMatrixTester):
    matrix_generator = None
    
    def generator(self):
        return self.matrix_generator
    
    ##########################################
    # Check that all matrix-matrix operations work
    ##########################################  
    generator_1 = generator
    generator_2 = generator
    def test_mul(self): self._test_mul()
    def test_matmul(self): self._test_matmul()
    def test_add(self): self._test_add()
    def test_sub(self): self._test_sub()

    
    def test_mul_scalar(self):
        for A,A_ref in self.generator().get_new_matrix(self):
            ###################################
            # multiply with scalar
            ###################################
            alpha = np.random.uniform(low=-10, high=10)
            alphaA     = alpha * A
            alphaA_ref = alpha * A_ref
            diff       = alphaA.toarray() - alphaA_ref
            tol        = np.finfo(diff.dtype).eps * np.linalg.norm(alphaA_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
            
    # def test_div_scalar(self):
    #     for A,A_ref in self.generator().get_new_matrix(self):
    #         ###################################
    #         # multiply with scalar
    #         ###################################
    #         alpha = np.random.uniform(low=-10, high=10)
    #         alphaA     = A / alpha
    #         alphaA_ref = A_ref / alpha
    #         diff       = alphaA.toarray() - alphaA_ref
    #         tol        = np.finfo(diff.dtype).eps * np.linalg.norm(alphaA_ref)
    #         assert np.linalg.norm(diff) <= self.get_tol(tol)
        
    def test_matmul_vector(self):
        for A,A_ref in self.generator().get_new_matrix(self):
            ###################################
            # multiply with dense vectors of shape (N,)
            ###################################
            x = np.random.uniform(low=-10, high=10, size=(A.shape[1],))
            Ax     = A     @ x
            Ax_ref = A_ref @ x
            diff   = Ax - Ax_ref
            tol    = np.finfo(diff.dtype).eps * np.linalg.norm(Ax_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
            assert Ax_ref.shape == Ax.shape
           
    def test_neg(self):
        for A,A_ref in self.generator().get_new_matrix(self):
            nA     = -A
            nA_ref = -A_ref
            diff   = nA.toarray() - nA_ref
            tol    = np.finfo(diff.dtype).eps * np.linalg.norm(nA_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
            
    def test_sum(self):
        for A,A_ref in self.generator().get_new_matrix(self):
            sumA = A.sum()
            sumA_ref = A_ref.sum()
            diff   = sumA - sumA_ref
            tol    = np.finfo(diff.dtype).eps * np.linalg.norm(sumA_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
            for axis in [0, 1]:
                sumA = A.sum(axis=axis)
                sumA_ref = A_ref.sum(axis=axis, keepdims=True)
                diff   = sumA.toarray() - sumA_ref
                tol    = np.finfo(diff.dtype).eps * np.linalg.norm(sumA_ref)
                assert np.linalg.norm(diff) <= self.get_tol(tol)
           
    def test_transpose(self):
        for A,A_ref in self.generator().get_new_matrix(self):
            AT     = A.transpose()
            AT_ref = A_ref.transpose()
            diff   = AT.toarray() - AT_ref
            tol    = np.finfo(diff.dtype).eps * np.linalg.norm(AT_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
            
    def test_abs(self):
        for A,A_ref in self.generator().get_new_matrix(self):
            absA     = abs(A)
            absA_ref = abs(A_ref)
            diff   = absA.toarray() - absA_ref
            tol    = np.finfo(diff.dtype).eps * np.linalg.norm(absA_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
    
    def test_asformat(self):
        formats = [None,'array','dense','coo','csr','csc','dia','dok','lil','bsr']
        
        for A,A_ref in self.generator().get_new_matrix(self):
            norm_A = np.linalg.norm(A_ref)
            for format in formats:
                diff = A.asformat(format) - A_ref
                tol = np.finfo(diff.dtype).eps * norm_A
                assert np.linalg.norm(diff) <= self.get_tol(tol)            
               
    def test_withformat(self):
        formats = ['dense','coo','csr','csc','dok','lil','bsr']
        
        for A,A_ref in self.generator().get_new_matrix(self):
            for format in formats:
                oldformat = A.format
                Awf_copy = A.withformat(format,copy=True)
                assert A.format == oldformat
                assert Awf_copy.format == [format for i in range(len(oldformat))]
                Awf = A.withformat(format,copy=False)
                assert A.format == Awf.format
                
        formats = [formats for i in range(len(oldformat))]
        for A,A_ref in self.generator().get_new_matrix(self):
            for format in list(itertools.product(*formats)):
                oldformat = A.format
                Awf_copy = A.withformat(format,copy=True)
                assert A.format == oldformat
                assert Awf_copy.format == list(format)
                Awf = A.withformat(format,copy=False)
                assert A.format == Awf.format
    
    def test_diagonal(self):
        for A,A_ref in self.generator().get_new_matrix(self,squared=True):
            d = A.diagonal()
            d_ref = A_ref.diagonal()
            diff   = d.toarray().reshape(-1) - d_ref
            tol    = np.finfo(diff.dtype).eps * np.linalg.norm(d_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
            D = kron.diag(d)
            D_ref = np.diag(d_ref)
            diff   = D.toarray() - D_ref
            tol    = np.finfo(diff.dtype).eps * np.linalg.norm(D_ref)
            assert np.linalg.norm(diff) <= self.get_tol(tol)
                
                

############################################################################################
############################################################################################
#
#   ASSAMBLE THE TEST CASES (and dissable specific parts)
#
############################################################################################
############################################################################################

KDIM = 2

############################################################
#
#   STANDALONE TESTS
#
#############################################################

###################################
# Kron
###################################
class Test_Kron(_IsolatedMatrixTester):
    matrix_generator = BasicKronMatrixGenerator_Large()
    kdim = KDIM
    
###################################
# Kronsum
###################################
class Test_KronSum(_IsolatedMatrixTester):
    matrix_generator = BasicKronSumMatrixGenerator()
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Not Implemented jet')
    def test_withformat(self): super().test_withformat()
    @pytest.mark.xfail(reason='abs for kronsum not possible')
    def test_abs(self): super().test_abs()
    
###################################
# Block
# ###################################
class Test_Block(_IsolatedMatrixTester):
    matrix_generator = BasicBlockMatrixGenerator()
    
    @pytest.mark.xfail(reason='Not Implemented jet')
    def test_withformat(self): super().test_withformat()
    
###################################
# Block of kron
###################################
class Test_KronBlock(_IsolatedMatrixTester):
    matrix_generator = KronBlockMatrixGenerator()
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Not Implemented jet')
    def test_withformat(self): super().test_withformat()


############################################################
#
#   INTERPLAY WITH DENSE
#
#############################################################

###################################
# Kron
###################################
class Test_Kron_w_Dense(_InterplayMatrixTester):
    matrix_generator_1 = BasicKronMatrixGenerator_Large()
    matrix_generator_2 = DenseMatrixGenerator()
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_add(self): super().test_add()
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_sub(self): super().test_sub()
    @pytest.mark.xfail(reason='Elementwise multiplication of None-Kron maticies not supported')
    def test_mul(self): super().test_mul()
    
###################################
# Kronsum
###################################
class Test_KronSum_w_Dense(_InterplayMatrixTester):
    matrix_generator_1 = BasicKronSumMatrixGenerator()
    matrix_generator_2 = DenseMatrixGenerator()
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_add(self): super().test_add()
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_sub(self): super().test_sub()
    @pytest.mark.xfail(reason='Elementwise multiplication of None-Kron maticies not supported')
    def test_mul(self): super().test_mul()

###################################
# Block
###################################
class Test_Block_w_Dense(_InterplayMatrixTester):
    matrix_generator_1 = BasicBlockMatrixGenerator()
    matrix_generator_2 = DenseMatrixGenerator()
    kdim = KDIM
    
###################################
# Block of kron
###################################
class Test_KronBlock_w_Dense(_InterplayMatrixTester):
    matrix_generator_1 = KronBlockMatrixGenerator()
    matrix_generator_2 = DenseMatrixGenerator()
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_add(self): super().test_add()
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_sub(self): super().test_sub()
    @pytest.mark.xfail(reason='Elementwise multiplication of None-Kron maticies not supported')
    def test_mul(self): super().test_mul()
    

############################################################
#
#   INTERPLAY WITH SPARSE
#
#############################################################

###################################
# Kron
###################################
class Test_Kron_w_Sparse(_InterplayMatrixTester):
    matrix_generator_1 = BasicKronMatrixGenerator_Large()
    matrix_generator_2 = SparseMatrixGenerator()
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_add(self): super().test_add()
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_sub(self): super().test_sub()
    @pytest.mark.xfail(reason='Elementwise multiplication of None-Kron maticies not supported')
    def test_mul(self): super().test_mul()
    
###################################
# Kronsum
###################################
class Test_KronSum_w_Sparse(_InterplayMatrixTester):
    matrix_generator_1 = BasicKronSumMatrixGenerator()
    matrix_generator_2 = SparseMatrixGenerator()
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_add(self): super().test_add()
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_sub(self): super().test_sub()
    @pytest.mark.xfail(reason='Elementwise multiplication of None-Kron maticies not supported')
    def test_mul(self): super().test_mul()
       
###################################
# Block
###################################
class Test_Block_w_Sparse(_InterplayMatrixTester):
    matrix_generator_1 = BasicBlockMatrixGenerator()
    matrix_generator_2 = SparseMatrixGenerator()
    matrix_generator_2.formats = ['csr','csc']
    kdim = KDIM
     
###################################
# Block of kron
###################################
class Test_KronBlock_w_Sparse(_InterplayMatrixTester):
    matrix_generator_1 = KronBlockMatrixGenerator()
    matrix_generator_2 = SparseMatrixGenerator()
    matrix_generator_2.formats = ['csr','csc']
    kdim = KDIM
    
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_add(self): super().test_add()
    @pytest.mark.xfail(reason='Addition of None-Kron maticies not supported')
    def test_sub(self): super().test_sub()
    @pytest.mark.xfail(reason='Elementwise multiplication of None-Kron maticies not supported')
    def test_mul(self): super().test_mul()
    
    
############################################################
#
#   INTERPLAY WITH KRON
#
#############################################################
    
###################################
# Kronsum
################################### 
class Test_KronSum_w_Kron(_InterplayMatrixTester):
    matrix_generator_1 = BasicKronSumMatrixGenerator()
    matrix_generator_2 = BasicKronMatrixGenerator()
    kdim = KDIM
    

if __name__ == "__main__":
    T = Test_Block_w_Sparse()
    T.test_mul()
    
    T = Test_KronSum()
    T.test_diagonal()
    
    T = Test_Block()
    T.test_diagonal()
    
    T = Test_KronBlock()
    T.test_diagonal()
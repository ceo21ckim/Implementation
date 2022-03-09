'''
Sparse Linear Regression
'''

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin 

class SparseLinearRegression(BaseEstimator, RegressorMixin):
    
    def __init__(self, n_iter=200, minval=1.0e-15, prune_mode=1, prune_threshold=1.0e-10, 
                 converge_min_iter=100, converge_threshold=1.0e-10, verbose=False, verbose_skip=10):
        self.n_iter = n_iter 
        self.minval = minval 
        self.prune_mode = prune_mode
        self.prune_threshold = prune_threshold
        self.converge_min_iter = converge_min_iter
        self.converge_threshold = converge_threshold
        self.verbose = verbose 
        self.verbose_skip = verbose_skip
        
        if verbose :
            print("SLiR (Sparse Liner Regression)")
            
    
    def fit(self, X, y):
        '''
        X : array-like, shape = (n_samples, n_features)
        y : array-like, shape = (n_samples)
        '''
        
        if y.ndim == 1:
            y = y.reshape(len(y), 1)
            
        
        # Check size 
        sample_num = X.shape[0]
        dim_num = X.shape[1]
        label_type_num = y.shape[1]
        if X.shape[0] != y.shape[0]:
            raise ValueError('The number of samples are unmatched betwwen x (%d) and y (%d).' % (X.shape[0], y.shape[0]))
        
        dim_num_org = dim_num
        
        # Transpose 
        X = X.transpose()
        Y = y.transpose()
        
        del y 
        
        if self.verbose:
            print('InputDim:%d/OutputLabels:%d/TrainNum:%d' % (dim_num, label_type_num, sample_num))
            
        # Initialization 
        
        X_var = np.mean(X ** 2, axis=1)
        Y_var = np.mean(Y ** 2, axis=1)
        
        # initialize alpha prior, weight, and noise variance 
        alpha_0 = 1.0 / np.mean(X_var)
        SY0 = np.mean(Y_var)
        
        if alpha_0 < self.minval:
            A = self.minval * np.ones((1, dim_num))
            
        else:
            A = alpha_0 * np.ones((1, dim_num))
            
        W = np.zeros((label_type_num, dim_num))
        SY = SY0 
        
        
        # prepare covariance of data and label
        YX = np.dot(Y, X.transpose())
        YY = np.sum(Y ** 2, axis=1)
        sumYY = np.sum(YY)
        
        # prepare covariance of X 
        
        if sample_num < dim_num:
            XX = None
            
        else:
            XX = np.dot(X, X.tranpose())
            
        # Not pruned index list 
        self.valid_index_list = np.arange(dim_num)
        
        # prepare temporary variables
        A_old = A[:, :]
        dim_num_old = dim_num 
        
        # Iterative procedure of ARDRegression
        # initial matrix를 만든 후 iteration을 수행하면서 학습함.
        for i in range(self.n_iter):
            if sample_num < dim_num: # sample_num : nrow / dim_num : ncol 
                XA = X.transpose() * A[0]
                CC = np.dot(XA, X) + np.eye(sample_num)
                XC = np.dot(X, np.linalg.pinv(CC))
                
                # Update weight
                W = YX * A[0]
                W = W - np.dot(np.dot(W, XC), XA)
                
                # Update gain
                G_A = A[0] * np.sum(X * XC, axis=1)
                
            else:
                # In the middle of iteration, this branch may be runned by the dimension reduction 
                if XX is None:
                    XX = np.dot(X, X.transpose())
                    
                    
                # Update weight 
                SW = XX + np.diag(1.0 / A[0])
                inv_SW = np.linalg.pinv(SW)
                W = np.dot(YX, inv_SW)
                
                # Update gain 
                G_A = np.diag(np.dot(XX, inv_SW)).transpose()
                
            # The sum of weight variance 
            WW = np.sum(W ** 2, axis=0)
            
            # Update noise variance 
            SY = (sumYY - np.sum(W * YX)) / (label_type_num * sample_num)
            
            if SY / SY0 < self.minval:
                dY = Y - np.dot(W, X)
                dYY = np.sum(dY ** 2, axis=1)
                SY = (np.sum(dYY) + np.sum(WW / A)) / (label_type_num * sample_num)
                SY = np.max([SY, self.minval])
                print('*')
                
            # Check Gain value
            G_A = np.maximum(G_A, self.minval)
            
            # Update alpha 
            try:
                A = np.sqrt(A * (WW / SY) / (G_A * label_type_num))
                
            except:
                print("Update error @ alpha")
                
            # Pruning dimensions : Pruning 은 가중치가 적게 부여된 값을 제거하면서 진행하는 모델을 의미함.
            if self.prune_mode > 0:
                # Prune the dimensions with the estimated precision of the weight over threshold
                if self.prune_mode == 1:
                    A_all = A[0] / np.max(A)
                    
                # Prune the dimensions with the weight variance over threshold
                elif self.prune_mode == 2:
                    A_all = WW / np.max(WW)
                    
                # Prune
                # 우리가 지정한 prune_threshold 보다 작으면 없애는 것을 의미함.
                activate_index = A_all > self.prune_threshold
                if np.sum(activate_index) < dim_num:
                    # Update dimension size 
                    dim_num_old = dim_num 
                    dim_num = np.sum(activate_index)
                    
                    # Reduce the dimensions
                    A = A[:, activate_index]
                    W = W[:, activate_index]
                    X = X[activate_index, :]
                    YX = YX[:, activate_index]
                    
                    # ixgrid = np.idx_([0, 1], [2, 4])
                    # ixgrid
                    # (array([[0], 
                    #         [1]]), array([[2, 4]]))

                    if XX is not None: # sparce case: sample_num < dim_num
                        XX = XX[np.ix_(activate_index, activate_index)]
                        
                    # Update valid index list
                    self.valid_index_list = self.valid_index_list[activate_index]
                    
                if len(self.valid_index_list) == 0 :
                    raise RuntimeError('ALL dimentions are pruned.')
                
                if self.verbose and (i + 1) % self.verbose_skip == 0 :
                    err = SY / SY0
                    print("Iter:%d, DimNum:%d, Error:%f" % (i + 1, dim_num, err))
                    
                # Check for convergence 
                if (i > self.converge_min_iter) & (dim_num == dim_num_old):
                    # Stop the iteration based on a max value of A diff 
                    Adif = np.max(np.abs(A - A_old))
                    if Adif < self.converge_threshold:
                        if self.verbose:
                            print("End. -- Iter:%d, DimNum:%d, Error:%f, AlphaConverged:%f" % (i, dim_num, err, Adif))
                            
                        break
                
                # Store A as A_old
                A_old = A[:, :]
            
            # Not pruning dimensions
            else:
                A = np.maximum(A, self.minval)
            
        self.__A = A
        self.__W = W
        
        # Copy sklearn's coef_, lambda_ and alpha_
        # Set 0 the pruned dimensions
        self.coef_ = np.zeros(dim_num_org)
        self.lambda_ = np.zeros(dim_num_org)
        
        # The coefficient of regression model
        self.coef_[self.valid_index_list] = self.__W
        
        # The estimated precisions of weights(coefficient)
        self.lambda_[self.valid_index_list] = self.__A
        self.alpha_ = SY 
        
        return self 
    
    def predict(self, X):
        # Transpose and reduce X
        X = X.transpose()[self.valid_index_list, :]
        
        # Predict by inner-producting and adding average of label 
        C = np.dot(self.__W, X)
        C = C.transpose().flatten()
        
        return C
import numpy as np 

class AlternativeLeastSquare:
    def __init__(self, rating, latent_dim=200, _alpha=40, r_lambda=40):
        self.R = rating
        self.latent_dim = latent_dim
        self.user_num, self.item_num = self.R.shape
        self.alpha = _alpha 
        self.r_lambda = r_lambda
        
        # initialization user, item matrix 
        self.X = np.random.randn(self.user_num, self.latent_dim) * 0.01 
        self.Y = np.random.randn(self.item_num, self.latent_dim) * 0.01
        
        self.xT = np.transpose(self.X)
        self.yT = np.transpose(self.Y)
        
        # preference matrix 
        P = self.R.copy()
        P[P>0] = 1
        self.P = P 
        
        # confidence matrix
        self.C = 1 + self.alpha * R 
        
    def loss_function(self):
        p_xT_y = np.square(self.P - np.matmul(self.X, self.yT))
        c_p_xT_y = np.sum(self.C * p_xT_y)
        r_x_y = self.r_lambda * (np.sum(np.square(self.X)) + np.sum(np.square(self.Y)))
        return np.sum(c_p_xT_y + r_x_y)
    
    def pred_X(self):
        '''
        self.yT : (latent_dim, item_num)
        self.Cu : (item_num, item_num)
        self.Y : (item_num, latent_dim)
        self.P[u,:] : (item_num, )
        
        yT_Cu_y : (latent_dim, latent_dim)
        yT_Cu_pu : (latent_dim, )
        
        r_identity : (latent_dim, latent_dim)
        '''
        for u in range(self.user_num):
            Cu = np.diag(self.C[u, :])
            yT_Cu_y = np.matmul(np.matmul(self.yT, Cu), self.Y) 
            yT_Cu_pu = np.matmul(np.matmul(self.yT, Cu), self.P[u,:]) 
            r_identity = self.r_lambda * np.identity(self.latent_dim)
            self.X[u] = np.linalg.solve(yT_Cu_y + r_identity, yT_Cu_pu)
            
    def pred_Y(self):
        
        '''
        self.xT : (latent_dim, user_num)
        self.Ci : (user_num, user_num)
        self.X : (user_num, latent_dim)
        self.P[:,i] : (user_num, )
        
        xT_Ci_x : (latent_dim, latent_dim)
        xT_Ci_pi : (latent_dim, )
        
        r_identity : (latent_dim, latent_dim)
        '''
        for i in range(self.item_num):
            Ci = np.diag(self.C[:, i])
            yT_Ci_y = np.matmul(np.matmul(self.xT, Ci), self.X)
            yT_Ci_pi = np.matmul(np.matmul(self.xT, Ci), self.P[:, i])
            r_identity = self.r_lambda * np.identity(self.latent_dim)
            self.Y[i] = np.linalg.solve(yT_Ci_y + r_identity, yT_Ci_pi)

       

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    R = np.array([
    [0, 0, 0, 0, 4, 2, 0, 1, 0, 0], 
    [1, 1, 0, 0, 3, 0, 2, 1, 4, 1], 
    [5, 5, 0, 3, 0, 0, 0, 0, 0, 1], 
    [0, 0, 0, 0, 4, 0, 0, 0, 5, 0], 
    [5, 6, 1, 0, 0, 7, 5, 0, 8, 0], 
    [0, 0, 5, 0, 0, 0, 0, 0, 0, 2], 
    [3, 3, 2, 1, 0, 0, 3, 0, 0, 0], 
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [5, 3, 2, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 3, 1], 
    [5, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    ])
    
    print('Matrix shape is ', R.shape)
    
    epochs = 15
    latent_dim = 200 
    r_lambda = 40
    _alpha = 40

    model = AlternativeLeastSquare(rating = R, latent_dim = latent_dim, _alpha = _alpha, r_lambda = r_lambda)
    loss = []
    for epoch in range(epochs):
        model.pred_X()
        model.pred_Y()
        
        loss.append(model.loss_function())
        print(f'model loss : {model.loss_function()}')
    
    plt.plot(np.arange(epoch+1), loss, label = 'ALS')
    plt.title('ALS LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


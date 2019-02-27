class BatchNorm(object):
    def __init__(self, fan_in, alpha):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        
        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x = x
        if eval == False:
            self.mean = np.mean(x, axis = 0)
            self.var = np.var(x, axis = 0)
            self.running_mean = self.alpha*self.running_mean + (1-self.alpha)*self.mean
            self.running_var = self.alpha*self.running_var + (1-self.alpha)*self.var
            self.norm = (x-self.mean)/np.sqrt(self.var + self.eps)
        else: 
            self.norm = (x-self.running_mean)/np.sqrt(self.running_var + self.eps)
        self.out = self.gamma*self.norm + self.beta

        return self.out

    def backward(self, delta):
        X_mean = self.x - self.mean
        temp_var = 1 / np.sqrt(self.var + self.eps)
        grad_norm = delta*self.gamma
        grad_var = -np.sum(grad_norm*X_mean, axis = 0)*0.5*(temp_var**3)
        grad_mu = -np.sum(grad_norm*temp_var, axis = 0) - grad_var*2.0*np.mean(X_mean, axis = 0) 
        
        self.dbeta = np.sum(delta, axis = 0)
        self.dgamma = np.sum(delta*self.norm, axis = 0)
        self.grad_x = (grad_norm*temp_var) + (2.0*grad_var*X_mean/self.x.shape[0]) + (grad_mu/self.x.shape[0])
        return self.grad_x
    
def random_normal_weight_init(d0, d1):
    return np.random.standard_normal(d0*d1).reshape((d0, d1))

def zeros_bias_init(d):
    return np.zeros((1, d))

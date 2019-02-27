class MLP(object):
    """ A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens,
                 activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum=0.0, num_bn_layers=0):
        
        # State parameters
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum

        self.W = []
        self.dW = []
        self.b = []
        self.db = []
        
        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = []
            
        # Input Output Parameters
        self.x = None
        self.y = None
        self.all_inputs = []
        self.velocity_W = []
        self.velocity_b = []
        
        all_layers = [[self.input_size], hiddens, [self.output_size]]
        
        self.all_layers = [element for subarray in all_layers for element in subarray]
        
        
        # Fill the weights matrices with initial values
        for layer in range(self.nlayers):
            self.W.append(np.array(weight_init_fn(self.all_layers[layer], self.all_layers[layer+1])))
            self.b.append(np.array(bias_init_fn(self.all_layers[layer+1])))
            self.velocity_W.append(np.zeros((self.all_layers[layer], self.all_layers[layer+1])))
            self.velocity_b.append(np.zeros(self.all_layers[layer+1]))
     
    # Forward pass of the MLP
    def forward(self, x):
        self.x = x
        prev_input = x
        for layer in range(self.nlayers):
            self.all_inputs.append(prev_input)
            output = np.matmul(prev_input, self.W[layer]) + self.b[layer]
            
            #Batch Norm
            if self.bn:
                if (layer < self.num_bn_layers):
                    self.bn_layers.append(BatchNorm(self.all_layers[layer+1], 0.9))
                    if (self.train_mode == True):
                        output = self.bn_layers[layer].forward(output)
                    else:
                        output = self.bn_layers[layer].forward(output, eval=True)
            
            self.activations[layer].forward(output)       
            prev_input = self.activations[layer].state
        return self.activations[layer].state

    # Make the gradients zero
    def zero_grads(self):
        self.dW = []
        self.db = []
        self.all_inputs = []

    # Step the gradients
    def step(self):
        if self.momentum:
            for layer in range(self.nlayers):
                self.velocity_W[layer] = self.momentum*(self.velocity_W[layer]) - self.lr*(self.dW[layer])
                self.velocity_b[layer] = self.momentum*(self.velocity_b[layer]) - self.lr*(self.db[layer])
                self.W[layer] += self.velocity_W[layer]
                self.b[layer] += self.velocity_b[layer]
        
        else:
            for layer in range(self.nlayers):
                self.W[layer] -= self.lr*self.dW[layer]
                self.b[layer] -= self.lr*self.db[layer]
        
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma -= self.lr*self.bn_layers[i].dgamma
                self.bn_layers[i].beta -= self.lr*self.bn_layers[i].dbeta

    # Backward pass
    def backward(self, labels):
        #Output Layer
        self.criterion(self.activations[self.nlayers-1].state, labels)
        smDer = self.criterion.derivative() #Softmax Derivative
        
        #Hidden Layers       
        grad_z = [] #z = output before Activation
        grad_y = [] #y = output after Activation
        grad_norm = [] #norm = batch normed output before Activation
        grad_y.append(smDer) 
        
        #Layers ahead of Batch Norm
        for i in range(self.nlayers - self.num_bn_layers):   
            grad_z.append(np.multiply(grad_y[i], self.activations[self.nlayers-i-1].derivative()))
            grad_y.append(np.matmul(grad_z[i], np.transpose(self.W[self.nlayers-i-1])))
            self.dW.append(np.matmul(np.transpose(self.all_inputs[self.nlayers-i-1]), grad_z[i])/self.x.shape[0])
            self.db.append(np.mean(grad_z[i], axis = 0))
       
        #Batch Norm
        for i in range(self.nlayers - self.num_bn_layers, self.nlayers):
            grad_norm.append(np.multiply(grad_y[i], self.activations[self.nlayers-i-1].derivative()))
            grad_z.append(self.bn_layers[self.nlayers-i-1].backward(grad_norm[i-(self.nlayers-self.num_bn_layers)])) 
            grad_y.append(np.matmul(grad_z[i], np.transpose(self.W[self.nlayers-i-1])))
            self.dW.append(np.matmul(np.transpose(self.all_inputs[self.nlayers-i-1]), grad_z[i])/self.x.shape[0])
            self.db.append(np.mean(grad_z[i], axis = 0))
        
        self.dW.reverse()
        self.db.reverse()
        
    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

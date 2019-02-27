class Identity(Activation):
    """ Identity function (already implemented).
     """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """ Implement the sigmoid non-linearity """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1/(1+np.exp(-x))
        return self.state

    def derivative(self):
        return self.state*(1-self.state)


class Tanh(Activation):
    """ Implement the tanh non-linearity """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        return self.state

    def derivative(self):
        return (1-(self.state)**2)


class ReLU(Activation):
    """ Implement the ReLU non-linearity """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        self.state[self.state < 0] = 0.0
        return self.state

    def derivative(self):
        der = self.state
        der[ der != 0] = 1
        return der

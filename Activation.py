class Activation(object):
    """ Interface for activation functions (non-linearities).

        In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        pass

    def derivative(self):
        pass

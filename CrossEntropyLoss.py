
class SoftmaxCrossEntropy(Criterion):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        loss = []
        max_elements = np.amax(self.logits, axis = 1).reshape((x.shape[0], 1))
        self.sm = (np.exp(self.logits-max_elements))/np.sum(np.exp(self.logits-max_elements), axis = 1).reshape((x.shape[0], 1))
        for element in -np.sum(self.labels*np.log(self.sm), axis = 1):
            loss.append(element)
        return loss
                           
    def derivative(self):
        return self.sm - self.labels

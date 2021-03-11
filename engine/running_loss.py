class RunningLoss:
    def __init__(self, momentum=0.9):
        self.loss = None
        self.momentum = momentum

    def __call__(self, new_loss: float):
        if self.loss is not None:
            self.loss = self.momentum * self.loss + new_loss * (1 - self.momentum)
        else:
            self.loss = new_loss

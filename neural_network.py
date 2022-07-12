from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super (NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # Flattens a contiguous range of dims into a tensor
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def predict(self, logits):
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)

        return y_pred
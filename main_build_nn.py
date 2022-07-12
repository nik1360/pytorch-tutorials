import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neural_network import NeuralNetwork
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


model = NeuralNetwork().to(device)
X = torch.rand(3, 28, 28, device=device)
logits = model(X) # NOTE: do not call model.forward(), just model()
y_pred = model.predict(logits)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
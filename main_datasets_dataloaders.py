import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#  Fashion-MNIST is a dataset of Zalando’s article images consisting
#  of 60,000 training examples and 10,000 test examples. Each example 
# comprises a 28×28 grayscale image and an associated label from one 
# of 10 classes.
training_data = datasets.FashionMNIST(
    root="data", # path where the data is stored 
    train=True, # specify train or test set
    download=True, # download data from internet if not availablein root
    transform=ToTensor() #specify transformation
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}



# plot a 3x3 grid with random images from the dataset
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# The Dataset retrieves our dataset’s features and labels one sample at a time. 
#  While training a model, we typically want to pass samples in “minibatches”, 
# reshuffle the data at every epoch to reduce model overfitting, and use 
# Python’s multiprocessing to speed up data retrieval.
# DataLoader is an iterable that abstracts this complexity for us in an easy API.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Each iteration below returns a batch of train_features and train_labels
#(containing batch_size=64 features and labels respectively). Because we 
# specified shuffle=True, after we iterate over all batches the data is 
# shuffled.

 
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
for i in range(0,10): # show 10 images of the mini batch
    img = train_features[i].squeeze()
    label = train_labels[i]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
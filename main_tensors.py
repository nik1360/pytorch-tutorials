import torch
import numpy as np

# Create a tensor from data
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# Create a tensor from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Tensor from another tensor 
x_ones = torch.ones_like(x_data) # retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data

shape = (2,3,) 
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

shape = rand_tensor.shape # get shape
data_type = rand_tensor.dtype # get type
dev = rand_tensor.device #get device

# Accessing tensor element
tensor = torch.ones(4, 4)
first_row = tensor[0]
first_column = tensor[:, 0]
last_column = tensor[..., -1]
tensor[:,1] = 0 # set the second column to zero

# Concatenating tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1) # dim=1 horizontal concatenation

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single element tensors
agg = tensor.sum() # aggregate all the calues of the tensor in a single element
agg_item = agg.item() # convert into a python numerical value 

# In place operation => store the result into the operand
tensor.add_(5)

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, 
# and changing one will change the other.
t = torch.ones(5)
n = t.numpy()
t.add_(1) # also affects n 

import torch 

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) 
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad) # derivative of the loss wrt W
print(b.grad) # derivative of the loss wrt b

# By default, all tensors with requires_grad=True are tracking their computational history 
# and support gradient computation. However, there are some cases when we do not need to do that, 
# for example, when we have trained the model and just want to apply it to some input data, 
# i.e. we only want to do forward computations through the network. We can stop tracking 
# computations by surrounding our computation code with torch.no_grad() block:

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# or usind detach()
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
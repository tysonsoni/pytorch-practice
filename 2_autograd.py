import torch

x = torch.randn(3, requires_grad=True)
y = x + 2
print(y)
z = y.mean()
print(z)
z.backward()  # calculate gradient dz/dx
print(x.grad)

w = y*y
print(f'w = {w}')
v = torch.tensor([1, 0.1, 0.01])
print(f'v = {v}')
w.backward(v)  # use vector if output is not scalar
print(f'x.grad = {x.grad}')

# deactivate autograd
x.requires_grad_(False)  # OR: y = x.detach()

# reset gradients to zero in a loop, otherwise they will add up
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() # reset gradients

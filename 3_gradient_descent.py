import torch
import torch.nn as nn

# Linear regression
# f = w * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.1
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate )

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = model(X)
    # loss
    l = loss(Y, y_pred)
    # calculate gradients = backward pass
    l.backward()
    # update weights
    optimizer.step()
    # zero the gradients after updating
    optimizer.zero_grad()
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

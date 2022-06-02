import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, variables):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(variables, 1)

    def forward(self, X):
        X = self.layer1(X)
        return X


def train_model(inputs, y, epochs=1000):
    criterion = nn.MSELoss()
    model = Model(inputs.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(epochs):
        optimizer.zero_grad()
        yhat = model(inputs)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
    return model


n = 10000

print('Example 3.2: Athletic performance')
x = torch.randn(n, 1)
z = 2 * x + torch.randn(n, 1)
features = torch.cat([x, z], 1)
y = 3 * z + torch.randn(n, 1)


model = train_model(features, y)

weights, bias = model.parameters()
print(weights.tolist())
print(bias.item())

model = train_model(x, y)

weights, bias = model.parameters()
print(weights.tolist())
print(bias.item())

print('\nExample 3.3: Competitiveness')
z = torch.randn(n, 1)  # athletic performance for n individuals
x = -2 * z + torch.randn(n, 1)  # preparation level for n individuals
# feature vector consists of preparation and athletic performance
features = torch.cat([x, z], 1)
y = x + 3 * z + torch.randn(n, 1)  # competitiveness level for n individuals


model = train_model(features, y)

weights, bias = model.parameters()
print(weights.tolist())
print(bias.item())

model = train_model(x, y)

weights, bias = model.parameters()
print(weights.tolist())
print(bias.item())

print('\nExample 3.4: Money')
x = torch.randn(n, 1)  # athletic performance for n individuals
y = torch.randn(n, 1)  # negotiating skill for n individuals
z = 2 * x + y + torch.randn(n, 1)  # salary for n individuals
# feature vector consists of athletic performance and salary
features = torch.cat([x, z], 1)


model = train_model(features, y)

weights, bias = model.parameters()
print(weights.tolist())
print(bias.item())

model = train_model(x, y)

weights, bias = model.parameters()
print(weights.tolist())
print(bias.item())

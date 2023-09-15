import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=512, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize the model
model = MLP()

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification (Accident YES/NO)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# Dummy training loop for 350 iterations
for epoch in range(350):
    # Here, load your saved features, for example:
    # inputs, labels = data

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every 50 iterations
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/350], Loss: {loss.item():.4f}")
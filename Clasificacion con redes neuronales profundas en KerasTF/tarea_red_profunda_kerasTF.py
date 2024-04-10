import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# da el conjunto de datos SVHN
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
test_data = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

# Crear dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Definir la red neuronal profunda
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            *[nn.Sequential(nn.Linear(3072 if i==0 else 100, 100), nn.ELU()) for i in range(20)],
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

# Instanciar la red
model = DNN()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Entrenar la red
for epoch in range(10):  # Número de épocas
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
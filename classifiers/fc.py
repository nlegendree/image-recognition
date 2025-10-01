import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix

class TwoLayerFC(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_fc_model(X_train, y_train, X_test, y_test, input_dim, epochs=30, lr=1e-3, batch_size=64, device="cpu"):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = TwoLayerFC(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device))
        y_pred = torch.argmax(preds, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

    return model, acc, cm, y_true, y_pred
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import gradio as gr
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int32)

# Împărțim automat 60.000 pentru antrenament, 10.000 pentru test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000, random_state=42, shuffle=True
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Etichetă: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
labels, counts = np.unique(y_train, return_counts=True)
freq_series = pd.Series(counts, index=labels)
print(freq_series)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("Valori pixel normalizate:", X_train.min(), "->", X_train.max())
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculăm acuratețea comparând predicțiile cu etichetele reale
acc = accuracy_score(y_test, y_pred)
print(f"Acuratețea pe setul de test: {acc:.4f}")

def predict_digit(img):
    try:
        img = img["composite"]
        img_pil = Image.fromarray(img).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        img_np = 255 - np.array(img_pil)
        img_flat = img_np.reshape(1, 784).astype('float32') / 255.0
        pred = model.predict(img_flat)[0]
        return f"Predicție Logistic Regression: {int(pred)}"
    except Exception as e:
        return f"Eroare: {str(e)}"

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000, random_state=42, shuffle=True
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 28, 28), dtype=torch.float32) / 255.0
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 28, 28), dtype=torch.float32) / 255.0
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_cnn = SimpleCNN()
import torch.optim as optim

device = "cpu"
model_cnn.to("cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

for epoch in range(5):
    model_cnn.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoca {epoch+1} – Loss: {running_loss / len(train_loader):.4f}")
    model_cnn.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Acuratețea CNN pe setul de test: {correct / total:.4f}")
# Matricea de confuzie
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.xlabel("Etichetă prezisă")
plt.ylabel("Etichetă reală")
plt.title("Matricea de confuzie - CNN pe MNIST")
plt.tight_layout()
plt.show()

def predict_digit_cnn(img):
    try:
        img = img["composite"]
        img_pil = Image.fromarray(img).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        img_np = 255 - np.array(img_pil)
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        model_cnn.eval()
        with torch.no_grad():
            output = model_cnn(img_tensor.to(device))
            prediction = torch.argmax(output, dim=1).item()

        return f"Predicție CNN: {prediction}"
    except Exception as e:
        return f"Eroare: {str(e)}"

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(),
    outputs="text",
    title="Recunoaștere cifre scrise de mână",
    description="Desenează o cifră și apasă Submit"
)

interface.launch()
"""
Binary (3 vs 6) MNIST classification.
"""

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import random
import sys
torch.set_default_dtype(torch.float64)
import utils
from utils import *

'''
cd public_test
python3 main.py
'''

# ---------------- hyper-parameters ----------------
N_COMPONENTS = 10
NUM_QUBITS = 4
BATCH_SIZE = 72
EPOCHS = 500  # 500
LR = 5e-4  # 5e-4
SEEDS = [0, 1, 2, 3, 4]
num_use_qubits = 3
ratio_use_data = 0.05
print_loss_period = 10
# --------------------------------------------------

elapsed_time_avg = 0
acc_avg = 0
for SEED in SEEDS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(".", train=True,  download=True, transform=transform)
    full_test  = datasets.MNIST(".", train=False, download=True, transform=transform)


    def extract_full(dataset):
        xs, ys = [], []
        for img, label in dataset:
            if label in (3, 6):
                xs.append(img.view(-1).numpy())
                ys.append(0 if label == 3 else 1)
        return np.stack(xs), np.array(ys)


    xs, ys = extract_full(full_train)
    print(f'전체 데이터 개수 {len(xs)} 3의 개수 {len(xs) - sum(ys)} 6의 개수 {sum(ys)}')
    xs, ys = extract_full(full_test)
    print(f'전체 테스트 데이터 개수 {len(xs)} 3의 개수 {len(xs) - sum(ys)} 6의 개수 {sum(ys)}')

    X_train_raw, y_train = extract(full_train, fraction=ratio_use_data)
    X_test_raw,  y_test  = extract(full_test, fraction=1.0)

    scaler = StandardScaler().fit(X_train_raw)
    X_train_std = scaler.transform(X_train_raw)
    X_test_std  = scaler.transform(X_test_raw)

    pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=SEED).fit(X_train_std)
    Xtr = pca.transform(X_train_std)
    Xte = pca.transform(X_test_std)

    scale = np.max(np.abs(Xtr))
    Xtr = (np.pi * Xtr[:, :4] / scale)
    Xte = (np.pi * Xte[:, :4] / scale)

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_train).double())
    test_ds  = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(y_test).double())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    dev = qml.device("lightning.qubit", wires=NUM_QUBITS)


    def feature_map(x):
        qml.templates.AngleEmbedding(x, wires=range(NUM_QUBITS), rotation="Y")


    def ansatz(params):
        qml.templates.StronglyEntanglingLayers(params, wires=range(NUM_QUBITS))


    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, params):
        qml.templates.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
        qml.templates.StronglyEntanglingLayers(params, wires=range(NUM_QUBITS))
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))


    NUM_LAYERS = 3  # circuit depth
    weight_shapes = {"params": (NUM_LAYERS, NUM_QUBITS, 3)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)


    class VQC(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.fc = nn.Sequential(
                nn.Linear(1, 1),
            )

        def forward(self, x):
            z = self.qlayer(x)[0]
            prob = (z + 1) / 2
            return prob.squeeze(-1)

    model = VQC()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    start_time = time.time()  # TRAIN START
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * Xb.size(0)
            total_correct += ((preds > 0.5) == yb).sum().item()

        avg_loss = total_loss / len(train_ds)
        acc = total_correct / len(train_ds)
        if epoch % print_loss_period == 0: print(f"Epoch {epoch:02d} - loss: {avg_loss:.4f}, acc: {acc:.4f}")
    end_time = time.time()  # TRAIN END
    elapsed_time = end_time - start_time

    # --- eval ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb)
            correct += ((preds > 0.5) == yb).sum().item()
    print(f"Test accuracy: {correct / len(test_ds):.4f}\n")

    elapsed_time_avg += elapsed_time
    acc_avg += correct / len(test_ds)
elapsed_time_avg /= len(SEEDS)
acc_avg /= len(SEEDS)

print()
# 1. 학습 속도
print(f"학습 속도: {elapsed_time_avg}")
# 2. 정확도
print(f"정확도: {acc_avg:.4f}")
# 3. 큐비트 사용률
print(f"큐비트 사용률: {num_use_qubits / NUM_QUBITS}")
# 4. 데이터 사용률
print(f"데이터 사용률: {ratio_use_data}")
# 5. 배치 크기
print(f"배치 크기: {BATCH_SIZE}")
# 6. 데이터 압축률
print(f"데이터 압축률: {784 / (N_COMPONENTS * 8)}")  # NUM_FEATURES = 10


# Result
# 학습 속도: 1229.188808441162
# 정확도: 0.9679
# 큐비트 사용률: 0.75
# 데이터 사용률: 0.05
# 배치 크기: 72
# 데이터 압축률: 9.8

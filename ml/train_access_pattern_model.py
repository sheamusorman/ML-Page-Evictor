from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from ml.access_pattern_model import AccessPatternClassifier

class AccessPatternDataset(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray):
    assert X.ndim == 2
    assert y.ndim == 1
    self.X = X.astype("float32")
    self.y = y.astype("int64")

  def __len__(self) -> int:
    return self.X.shape[0]

  def __getitem__(self, idx: int):
    x = torch.from_numpy(self.X[idx])
    y = torch.tensor(self.y[idx], dtype=torch.long)
    return x, y


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
  """
  Compute inverse-frequency class weights for cross-entropy.
  """
  classes, counts = np.unique(y, return_counts=True)
  num_classes = int(classes.max()) + 1

  freq = np.zeros(num_classes, dtype=np.float32)
  freq[classes] = counts.astype(np.float32)

  # Inverse frequency
  eps = 1e-6
  inv = 1.0 / (freq + eps)
  inv /= inv.mean()

  return torch.tensor(inv, dtype=torch.float32)


def train_model(
  X: np.ndarray,
  y: np.ndarray,
  epochs: int = 40,
  batch_size: int = 64,
) -> AccessPatternClassifier:
  dataset = AccessPatternDataset(X, y)

  n_total = len(dataset)
  n_train = int(0.8 * n_total)
  n_val = n_total - n_train
  train_ds, val_ds = random_split(dataset, [n_train, n_val])

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

  input_dim = X.shape[1]
  num_classes = int(y.max()) + 1

  model = AccessPatternClassifier(input_dim=input_dim, num_classes=num_classes)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  class_weights = compute_class_weights(y)

  for epoch in range(epochs):
    # Training
    model.train()
    for xb, yb in train_loader:
      logits = model(xb)
      loss = F.cross_entropy(logits, yb, weight=class_weights)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for xb, yb in val_loader:
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))
    acc = correct / total if total > 0 else 0.0
    print(f"epoch {epoch:02d}  val_acc={acc:.3f}")

  return model


if __name__ == "__main__":
  X = np.load("X_access_pattern.npy")
  y = np.load("y_access_pattern.npy")
  print("Loaded X_access_pattern.npy and y_access_pattern.npy")

  model = train_model(X, y, epochs=40, batch_size=64)
  torch.save(model.state_dict(), "access_pattern_classifier.pt")
  print("Saved model to access_pattern_classifier.pt")

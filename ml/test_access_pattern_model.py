from __future__ import annotations

import numpy as np
import torch

from ml.access_pattern_model import AccessPatternClassifier

ACCESS_PATTERN_LABELS = {
  0: "A (scan/stride)",
  1: "B (random)",
  2: "C (hot set)",
  3: "D (loop)",
  4: "E (phase shift)",
  5: "F (mixed)",
}

def main() -> None:
  # Load dataset
  X = np.load("X_access_pattern.npy").astype("float32")
  y = np.load("y_access_pattern.npy").astype("int64")

  num_samples, input_dim = X.shape
  num_classes = int(y.max()) + 1

  # Load model
  model = AccessPatternClassifier(input_dim=input_dim, num_classes=num_classes)
  state = torch.load("access_pattern_classifier.pt", map_location="cpu")
  model.load_state_dict(state)
  model.eval()

  # Run inference
  with torch.no_grad():
    xb = torch.from_numpy(X)
    logits = model(xb)
    preds = logits.argmax(dim=1).numpy()

  # Overall accuracy
  correct = (preds == y).sum()
  acc = correct / num_samples if num_samples > 0 else 0.0
  print(f"Overall accuracy: {acc:.3f} ({correct}/{num_samples})")

  # Per-class accuracy
  print("\nPer-class accuracy:")
  for c in range(num_classes):
    mask = (y == c)
    n_c = int(mask.sum())
    if n_c == 0:
      print(f"  class {c}: (no samples)")
      continue
    correct_c = int((preds[mask] == y[mask]).sum())
    acc_c = correct_c / n_c
    label_name = ACCESS_PATTERN_LABELS.get(c, f"class {c}")
    print(f"  {label_name}: {acc_c:.3f} ({correct_c}/{n_c})")

  # Confusion matrix
  print("\nConfusion matrix (rows = true, cols = pred):")
  conf = np.zeros((num_classes, num_classes), dtype=np.int64)
  for t, p in zip(y, preds):
    conf[t, p] += 1

  # Confusion matrix with headers
  header = "      " + " ".join([f"{c:5d}" for c in range(num_classes)])
  print(header)
  for i in range(num_classes):
    row_vals = " ".join([f"{conf[i, j]:5d}" for j in range(num_classes)])
    print(f"{i:3d}: {row_vals}")


if __name__ == "__main__":
  main()

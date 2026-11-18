import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from minigrad.nn import MLP
from minigrad.optim import SGD


random.seed(42)

X, y = make_moons(
    n_samples=100,
    noise=0.1,
    random_state=42
)
y = np.where(y == 0, -1.0, 1.0)
model = MLP(2, [16, 16, 1])
optimizer = SGD(model.parameters(), lr=0.05)

losses = []

for epoch in range(100):
    predictions = [model(x) for x in X]

    loss = sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, y)
    ) / len(y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.data)

    if epoch % 10 == 0:
        correct = sum(
            (prediction.data > 0) == (target > 0)
            for prediction, target in zip(predictions, y)
        )

        accuracy = correct / len(y)

        print(
            f"epoch {epoch:3d} | "
            f"loss {loss.data:.4f} | "
            f"accuracy {accuracy:.2%}"
        )
        
predictions = [model(x) for x in X]

correct = sum(
    (prediction.data > 0) == (target > 0)
    for prediction, target in zip(predictions, y)
)

final_accuracy = correct / len(y)

print(f"\nFinal loss: {losses[-1]:.4f}")
print(f"Final accuracy: {final_accuracy:.2%}")

plt.figure(figsize=(8, 5))

plt.plot(range(len(losses)), losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MiniGrad Training Loss on Two Moons")

plt.tight_layout()
plt.savefig("images/loss_curve.png", dpi=150)
plt.close()

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 150),
    np.linspace(y_min, y_max, 150)
)

grid = np.c_[xx.ravel(), yy.ravel()]

predictions = np.array([
    model(point).data for point in grid
])

predictions = predictions.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    predictions,
    levels=[-1, 0, 1],
    alpha=0.3
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("MiniGrad Decision Boundary on Two Moons")

plt.tight_layout()
plt.savefig("images/decision_boundary.png", dpi=150)
plt.close()
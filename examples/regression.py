import random

import matplotlib.pyplot as plt
import numpy as np

from minigrad.nn import MLP
from minigrad.optim import SGD


random.seed(42)

X = np.linspace(-2, 2, 50)
y = X**2

model = MLP(
    1,
    [16, 16, 1],
    activation="tanh",
    output_activation="linear"
)

optimizer = SGD(model.parameters(), lr=0.01)

losses = []

for epoch in range(200):
    predictions = [model([x]) for x in X]

    loss = sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, y)
    ) / len(y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.data)

    if epoch % 20 == 0:
        print(
            f"epoch {epoch:3d} | "
            f"loss {loss.data:.4f}"
        )

predictions = np.array([
    model([x]).data for x in X
])

print(f"\nFinal loss: {losses[-1]:.4f}")

plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Target")
plt.plot(X, predictions, label="MiniGrad Prediction")

plt.xlabel("x")
plt.ylabel("y")
plt.title("MiniGrad Regression: Learning y = x²")
plt.legend()

plt.tight_layout()
plt.savefig("images/regression_fit.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))

plt.plot(range(len(losses)), losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MiniGrad Regression Training Loss")

plt.tight_layout()
plt.savefig("images/regression_loss.png", dpi=150)
plt.close()
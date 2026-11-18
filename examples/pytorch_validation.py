import torch

from minigrad.engine import Value


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

y = (a * b + c).tanh()
y.backward()


a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
b_torch = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
c_torch = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)

y_torch = torch.tanh(a_torch * b_torch + c_torch)
y_torch.backward()


print("Output")
print("MiniGrad:", y.data)
print("PyTorch: ", y_torch.item())

print("\nGradient da")
print("MiniGrad:", a.grad)
print("PyTorch: ", a_torch.grad.item())

print("\nGradient db")
print("MiniGrad:", b.grad)
print("PyTorch: ", b_torch.grad.item())

print("\nGradient dc")
print("MiniGrad:", c.grad)
print("PyTorch: ", c_torch.grad.item())
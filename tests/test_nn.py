import random
from minigrad.nn import Neuron, Layer, MLP
from minigrad.optim import SGD

def test_neuron_parameters():
    n = Neuron(3)

    assert len(n.parameters()) == 4


def test_layer_output():
    layer = Layer(3, 4)
    output = layer([1.0, 2.0, 3.0])

    assert len(output) == 4


def test_mlp_output():
    model = MLP(3, [4, 4, 1])
    output = model([1.0, 2.0, 3.0])

    assert hasattr(output, "data")


def test_mlp_parameters():
    model = MLP(3, [4, 4, 1])

    assert len(model.parameters()) == 41


def test_zero_grad():
    model = MLP(3, [4, 1])
    optimizer = SGD(model.parameters(), lr=0.01)

    for p in model.parameters():
        p.grad = 2.0

    optimizer.zero_grad()

    for p in model.parameters():
        assert p.grad == 0.0


def test_sgd_step():
    model = MLP(3, [1])
    optimizer = SGD(model.parameters(), lr=0.1)

    p = model.parameters()[0]
    old_value = p.data

    p.grad = 2.0
    optimizer.step()

    assert abs(p.data - (old_value - 0.2)) < 1e-6

def test_network_learns():
    random.seed(42)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])
    optimizer = SGD(model.parameters(), lr=0.05)

    ypred = [model(x) for x in xs]
    initial_loss = sum((ypred - ygt)**2 for ypred, ygt in zip(ypred, ys)).data

    for _ in range(100):
        ypred = [model(x) for x in xs]
        loss = sum((ypred - ygt)**2 for ypred, ygt in zip(ypred, ys))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ypred = [model(x) for x in xs]
    final_loss = sum((ypred - ygt)**2 for ypred, ygt in zip(ypred, ys)).data

    assert final_loss < initial_loss
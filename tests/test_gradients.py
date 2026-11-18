import math

from minigrad.engine import Value


def numerical_gradient(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()

    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()

    assert c.data == 6.0
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_power():
    a = Value(3.0)
    c = a ** 2
    c.backward()

    assert c.data == 9.0
    assert a.grad == 6.0


def test_subtraction():
    a = Value(5.0)
    b = Value(2.0)
    c = a - b
    c.backward()

    assert c.data == 3.0
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_division():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    c.backward()

    assert c.data == 3.0
    assert abs(a.grad - 0.5) < 1e-6
    assert abs(b.grad + 1.5) < 1e-6


def test_exp():
    a = Value(2.0)
    c = a.exp()
    c.backward()

    assert abs(c.data - math.exp(2.0)) < 1e-6
    assert abs(a.grad - math.exp(2.0)) < 1e-6


def test_tanh():
    a = Value(0.5)
    c = a.tanh()
    c.backward()

    expected = math.tanh(0.5)

    assert abs(c.data - expected) < 1e-6
    assert abs(a.grad - (1 - expected**2)) < 1e-6


def test_relu_positive():
    a = Value(3.0)
    c = a.relu()
    c.backward()

    assert c.data == 3.0
    assert a.grad == 1.0


def test_relu_negative():
    a = Value(-3.0)
    c = a.relu()
    c.backward()

    assert c.data == 0.0
    assert a.grad == 0.0


def test_relu_zero():
    a = Value(0.0)
    c = a.relu()
    c.backward()

    assert c.data == 0.0
    assert a.grad == 0.0


def test_chained_expression():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a**2
    c.backward()

    assert c.data == 10.0
    assert a.grad == 7.0
    assert b.grad == 2.0


def test_gradient_accumulation():
    a = Value(3.0)
    c = a * a
    c.backward()

    assert a.grad == 6.0


def test_numerical_gradient():
    x = Value(2.0)
    y = x**3 + 2*x
    y.backward()

    def f(x):
        return x**3 + 2*x

    numerical_grad = numerical_gradient(f, 2.0)

    assert abs(x.grad - numerical_grad) < 1e-5
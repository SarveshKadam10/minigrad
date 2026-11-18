import random

from .engine import Value

class Neuron:

  def __init__(self,nin,activation = "tanh"):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.activation = activation

  def __call__(self,x):
    act = sum((wi*xi for wi, xi in zip(self.w,x)),self.b)

    if self.activation == "tanh":
      out = act.tanh()
    elif self.activation == "relu":
      out = act.relu()
    elif self.activation == "linear":
      out = act
    else:
      raise ValueError(f"Unknown activation: {self.activation}")
    
    return out

  def parameters(self):
    p = self.w + [self.b]
    return p

class Layer:

  def __init__(self,nin,nout, activation="tanh"):
    self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

  def __call__(self,x):
    outs = [neuron(x) for neuron in self.neurons]
    return outs[0] if len(outs)==1 else outs

  def parameters(self):
    p = [p for neuron in self.neurons for p in neuron.parameters()]
    return p

class MLP:

  def __init__(self,nin,nouts, activation="tanh", output_activation="tanh"):
    sz = [nin] + nouts
    self.layers = []
    for i in range(len(nouts)):
      layer_activation = (output_activation if i == len(nouts) - 1 else activation)
      self.layers.append(Layer(sz[i],sz[i+1], activation=layer_activation))

  def __call__(self,x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    p = [p for layer in self.layers for p in layer.parameters()]
    return p
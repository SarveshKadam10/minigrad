from minigrad.engine import Value
from minigrad.visualize import draw_graph


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = d.tanh()
f.label = "f"

f.backward()

draw_graph(f)
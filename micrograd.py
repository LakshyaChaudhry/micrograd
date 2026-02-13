#This is my replication of Andrej Karpathy's micrograd project to understand how neural networks work, and are built from scratch.
#There is no AI or auto complete used in this project.

import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph # for the visualization of the graph


def f(x):
    return 3*x**2 - 4*x + 5 #copied function from the video

xs = np.arange(-5,5,0.25)
ys = f(xs)
# plt.plot(xs,ys)
# plt.show()

h = 0.0001
x = 3.0

# df = (f(x + h) - f(x)) / h
# print(df)

#Now ill implement the more complex mathematical function


# d1 = a*b + c
# a += h
# d2 = a*b + c

# slope = (d2 - d1) / h
# print(slope)

class value:

    def __init__(self, data, _children=(), _op='', _label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._label = _label
        self.grad = 0.0
    
    def __repr__(self):
        return self.data

    def __add__(self, other):
        out = value(self.data + other.data, (self, other), _op='+')
        return out

    def __mul__(self, other):
        out = value(self.data * other.data, (self, other), _op='*')
        return out


a = value(2.0); a._label='a'
b = value(-3.0); b._label='b'
c = value(10.0); c._label='c'
e = a*b; e._label='e'
d = e + c; d._label='d'

#At this point, the forward pass is complete + the graph is visualized.
# next is implementing backpropagation.

#manual backpropogration below:
d.grad = 1.0
e.grad = 1.0
c.grad = 1.0
b.grad = 2.0
a.grad = -3.0


#Code for visualization of the graph is below:

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n._label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

# Create and display the visualization like a matplotlib plot (extra logic added since we aren't in a notebook):
graph = draw_dot(d)
from PIL import Image
graph.format = 'png'
graph.render('temp_graph', cleanup=True)
img = Image.open('temp_graph.png')
plt.figure(figsize=(12, 4))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()


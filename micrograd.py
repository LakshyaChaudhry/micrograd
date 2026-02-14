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
        self._backward = lambda: None
        self.grad = 0.0
    
    def __repr__(self):
        return self.data

    def __add__(self, other):
        out = value(self.data + other.data, (self, other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = value(self.data * other.data, (self, other), _op='*')
        def _backward():
          self.grad += other.data * out.grad
          other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
      return self * other
    
    def exp(self):
      x = self.data
      out = value(math.exp(x), (self, ), _op='exp')

      def _backward():
        self.grad += out.data * out.grad
      
      out._backward = _backward
      return out
    
    def __truediv__(self, other):
      return self * other**-1
    
    def __pow__(self, other):
      assert isinstance(other, (int, float)), "only supporting int/float powers for now"
      out = value(self.data**other, (self, ), _op=f'**{other}')

      def _backward():
        self.grad += other * (self.data**(other-1)) * out.grad
      out._backward = _backward
      return out

    def __neg__(self):
      return self * -1
    
    def __sub__(self, other):
      return self + (-other)

    def tanh(self):
      n = self.data
      t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
      out = value(t, (self, ), _op='tanh')

      def _backward():
        self.grad += (1-t**2) * out.grad
      
      out._backward = _backward
      return out

    def backwards(self):
      topo = []
      visited = set()
      def build_topo(v):
        if v not in visited:
          visited.add(v)
          for child in v._prev:
            build_topo(child)
          topo.append(v)
      build_topo(self)

      self.grad = 1.0
      for node in reversed(topo):
        node._backward()




#implementing the neuron below: 
x1 = value(2.0); x1._label='x1'
x2 = value(0.0); x2._label='x2'
w1 = value(-3.0); w1._label='w1'
w2 = value(1.0); w2._label='w2'

b = value(6.881373587019543); b._label='b' #bias value from the video
x1w1 = x1*w1; x1w1._label='x1*w1'
x2w2 = x2*w2; x2w2._label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2._label='x1w1 + x2w2'
n = x1w1x2w2 + b; n._label='n'
o = n.tanh(); o._label='o'

o.backwards()

#now we need to automate the backpropagation process.






















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
graph = draw_dot(o)
from PIL import Image
graph.format = 'png'
graph.render('temp_graph', cleanup=True)
img = Image.open('temp_graph.png')
plt.figure(figsize=(12, 4))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()


from math import log, exp, tanh

# Node in computational graph
# Stores operations, operands, data, and gradients.
# Computes gradients via Backprop.
class Value:
    def __init__(self, data=None, children=(), label=None):
        self.data = data
        self.children = set(children)
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self.op = ''

    def __repr__(self):
        return "Value(data={})".format(self.data)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmull__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-1 * other)

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1
    
    def __pow__(self, other):
        # hopefully other is a float
        out = Value(self.data ** other, (self, ), '**')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        
        out._backward = _backward
        return out

    def exp(self):
        x = exp(self.data)
        out = Value(x, (self, ), 'exp')

        def _backward():
            self.grad += x * out.grad
        
        out._backward = _backward
        return out

    def log(self):
        x = log(self.data)
        out = Value(x, (self, ), 'log')

        def _backward():
            log_grad = (1.0 / self.data)
            self.grad += log_grad * out.grad
        
        out._backward = _backward
        return out

    def tanh(self):
        x = tanh(self.data)
        out = Value(x, (self, ), 'tanh')

        def _backward():
            tanh_grad = 1 - x**2
            self.grad += tanh_grad * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        # Topological Sort
        visited = set()
        result = []

        def topo_sort(node):
            # given a node, add it to the result if all dependencies have been added to the list
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    topo_sort(child)
                result.append(node)

        topo_sort(self)

        # Call backward in reverse to fill in gradients.
        self.grad = 1.0
        for node in reversed(result):
            node._backward()
    
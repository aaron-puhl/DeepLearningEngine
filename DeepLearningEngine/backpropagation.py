
class Value:
    def __init__(self, value, _prev = [], _backward = lambda x: x):
        self._value = value
        self._prev = _prev
        self._backward = lambda : _backward(self)
        self.gradient = 0.0

    def __repr__(self):
        return f"Value({self._value})"

    def __add__(self, other):
        x = self._value + other._value

        def _backward(out):
            self.gradient += 1 * out.gradient
            other.gradient += 1 * out.gradient

        return Value(x, (self,other), _backward)

    def __sub__(self, other):
        x = self._value - other._value

        def _backward(out):
            self.gradient += 1 * out.gradient
            other.gradient += -1 * out.gradient

        return Value(x, (self,other), _backward)

    def __mul__(self, other):
        x = self._value * other._value

        def _backward(out):
            self.gradient += other._value * out.gradient
            other.gradient += self._value * out.gradient

        return Value(x, (self,other), _backward)

    def __truediv__(self, other):
        return self * other**(-1)

    def __pow__(self, exponent):
        x = self._value ** exponent

        def _backward(out):
            self.gradient += exponent * self._value ** (exponent - 1) * out.gradient
        
        return Value(x, [self], _backward)

    def __neg__(self):
        return Value(0) - self

    def backwards(self):
        self.gradient = 1.0
        sorted = []
        def sort_topological(current):
            if current not in sorted:
                for child in current._prev:
                    sort_topological(child)
                sorted.append(current)
        sort_topological(self)

        for item in reversed(sorted):
            item._backward()

    def zero_grad(self):
        self.gradient = 0.0
        for x in self._prev:
            x.zero_grad()

    def leaky_relu(self):
        if self._value >= 0:
            return self * Value(1)
        else:
            return self * Value(0.1)

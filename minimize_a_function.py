"""
Idea is that we use library to minimize
a function, sat x**2
"""
from autograd.tensor import Tensor

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)


# minimize the sum of squares
for i in range(100):

    x.zero_grad()
    
    sum_of_squares = (x * x).sum() # is a 0-tensor
    sum_of_squares.backward()

    delta_x = 0.1 * x.grad
    x -= delta_x  # Used this after defining inplace __imul__ operation on tensors
    
    #x = Tensor(x.data - delta_x.data, requires_grad=True)

    print(i, sum_of_squares)


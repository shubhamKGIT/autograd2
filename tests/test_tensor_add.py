import unittest
import pytest

from autograd.tensor import Tensor, add


class TestTensorsum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = add(t1, t2)

        t3.backward(Tensor([-1, -2, -3]))

        assert t1.grad.data.tolist() == [-1, -2, -3]
        assert t2.grad.data.tolist() == [-1, -2, -3]

    def test_broadcast_add(self):
        # Broadcasting?
        # Add 1s to beginning of each shape when multiplying two matrices
        # Default is t1*t2 when t1.shape, t2.shape is given 

        # t1.shape ==(10,5), t2.shape==(5,) => t1+t2, t2 viewed as (1, 5)

        # Second thing: If t2 has one dimension extra
        # We can add t1 as one row to t2

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([7, 8, 9], requires_grad=True)

        t3 = add(t1, t2)

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1],[1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)

        t3 = add(t1, t2)

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1],[1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]
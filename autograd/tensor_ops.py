from typing import List

import numpy as np

from autograd.tensor import Tensor, Dependency


def _tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and resutnr the 0-tensor
    that's the summ of all elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so 
            each input elements contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    
    else:
        depends_on = []


    return Tensor(data,
                  requires_grad,
                  depends_on)

def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Idea: [1, 2, 3] + [2+e, 3, 4] = [3+e, 5, 7]
            # Every change in one of the tensors is 
            # reflected in the sum

            # To handle broadcasting properly 
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i , dim in enumerate(t1.shape):
                if dim ==1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad    # return whatever is the gradient with tensors
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            
            #### Handling broadcasting ####
            # Sum across added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i , dim in enumerate(t2.shape):
                if dim ==1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad 

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    y = a * b 
    and we change a by small amount eps 
    y = (a + eps) * b = (a * b) + (eps * b)

    say have dL/dy
    wanna find dL/da = dL/dy * dy/da
    """
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data

            # To handle broadcasting properly 
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i , dim in enumerate(t1.shape):
                if dim ==1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad    # return whatever is the gradient with tensors
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            
            grad = grad * t1.data
            #### Handling broadcasting ####
            # Sum across added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i , dim in enumerate(t2.shape):
                if dim ==1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad 

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad

    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on =[]

    return Tensor(data, requires_grad, depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    # return _add(t1, _neg(t2))
    return t1 + -t2

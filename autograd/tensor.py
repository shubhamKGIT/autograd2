from typing import List, NamedTuple, Callable, Optional, Union

import numpy as np



class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # setting the data manually means we invalidate the gradient
        self.grad = None
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """ 
        Gets called if one does t + other
        """
        return _add(self, ensure_tensor(other))
        # Made this add function private

    def __radd__(self, other) -> 'Tensor':
        """
        Gets called if one does other + t
        """
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        """ when we do t+= other"""
        self.data = self.data + ensure_tensor(other).data
        # need not invalidate grad, will happen automatically
        return self
    
    def __isub__(self, other) -> 'Tensor':
        """ when we do t-= other"""
        self.data = self.data - ensure_tensor(other).data
        # invalidate the gradient 
        self.grad = None

        return self

    def __imul__(self, other) -> 'Tensor':
        """ when we do t*= other"""
        self.data = self.data * ensure_tensor(other).data
        # invalidate the gradient 
        self.grad = None

        return self

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return _mul(ensure_tensor(other), self)

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, other)
        

    def __neg__(self) -> 'Tensor':
        return _neg(self)
    
    def __sub__(self, other) -> 'Tensor':
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return _sub(ensure_tensor(other), self)

    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)


    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad-tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be  specified for non-zero tensor")
        
        self.grad.data = self.grad.data + grad.data # type:ignore

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return _tensor_sum(self)

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

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    If t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad is (n1, m2)
    if t3 = t1 @ t2 adn grad3 is gradient of some function wrt t3, 
    then 
    grad1 = grad @ t2.T
    grad2 = t1.T @ grad
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            #grad = grad * t2.data
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            #grad = grad * t1.data
            return t1.data.T @ grad
            
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)
    

def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad 
            return bigger_grad

        depends_on = Dependency(t, grad_fn)

    else:
        depends_on =[]

    return Tensor(data, requires_grad, depends_on)


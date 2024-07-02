import autograd as ag
import numpy as np

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [x.numpy() for x in out.op.gradient(ag.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_add_backward():
    gradient_check(ag.add, ag.Tensor(np.random.randn(5, 4)), ag.Tensor(np.random.randn(5, 4)))
    print('Pass add backward!')


def test_add_scaler_backward():
    gradient_check(ag.add_scalar, ag.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1))
    print('Pass add_scalar backward!')


def test_multiply_backward():
    gradient_check(ag.multiply, ag.Tensor(np.random.randn(5, 4)), ag.Tensor(np.random.randn(5, 4)))
    print('Pass multiply backward!')


def test_mul_scaler_backward():
    gradient_check(ag.mul_scalar, ag.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1))
    print('Pass mul_scalar backward!')


def test_power_scaler_backward():
    gradient_check(ag.power_scalar, ag.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1))
    print('Pass power_scalar backward!')


def test_divide_backward():
    gradient_check(ag.divide, ag.Tensor(np.random.randn(5, 4)), ag.Tensor(5 + np.random.randn(5, 4)))
    print('Pass divide_backward!')


def test_divide_scalar_backward():
    gradient_check(ag.divide_scalar, ag.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1))
    print('Pass divide_scalar backward!')


def test_transpose_backward():
    gradient_check(ag.transpose, ag.Tensor(np.random.randn(3, 5, 4)), dim=(1, 2))
    gradient_check(ag.transpose, ag.Tensor(np.random.randn(3, 5, 4)), dim=(0, 1))
    print('Pass transpose backward!')


def test_reshape_backward():
    gradient_check(ag.reshape, ag.Tensor(np.random.randn(5, 4)), shape=(4, 5))
    print('Pass reshape backward!')


def test_broadcast_to_backward():
    gradient_check(ag.broadcast_to, ag.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(ag.broadcast_to, ag.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(ag.broadcast_to, ag.Tensor(np.random.randn(1,)), shape=(3, 3, 3))
    gradient_check(ag.broadcast_to, ag.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(ag.broadcast_to, ag.Tensor(np.random.randn(5, 4, 1)), shape=(5, 4, 3))
    print('Pass broadcast_to backward!')


def test_summation_backward():
    gradient_check(ag.summation, ag.Tensor(np.random.randn(5, 4)), dim=(1,))
    gradient_check(ag.summation, ag.Tensor(np.random.randn(5, 4)), dim=(0,))
    gradient_check(ag.summation, ag.Tensor(np.random.randn(5, 4)), dim=(0, 1))
    gradient_check(ag.summation, ag.Tensor(np.random.randn(5, 4, 1)), dim=(0, 1))
    print('Pass summation backward!')


def test_matmul_simple_backward():
    gradient_check(ag.matmul, ag.Tensor(np.random.randn(5, 4)), ag.Tensor(np.random.randn(4, 5)))
    print('Pass matmul_simple backward!')


def test_matmul_batched_backward():
    gradient_check(ag.matmul, ag.Tensor(np.random.randn(6, 6, 5, 4)), ag.Tensor(np.random.randn(6, 6, 4, 3)))
    gradient_check(ag.matmul, ag.Tensor(np.random.randn(6, 6, 5, 4)), ag.Tensor(np.random.randn(6, 4, 3)))
    gradient_check(ag.matmul, ag.Tensor(np.random.randn(6, 6, 5, 4)), ag.Tensor(np.random.randn(4, 3)))
    gradient_check(ag.matmul, ag.Tensor(np.random.randn(5, 4)), ag.Tensor(np.random.randn(6, 6, 4, 3)))
    gradient_check(ag.matmul, ag.Tensor(np.random.randn(6, 5, 4)), ag.Tensor(np.random.randn(6, 6, 4, 3)))
    print('Pass matmul_batched_backward!')


def test_negate_backward():
    gradient_check(ag.negate, ag.Tensor(np.random.randn(5, 4)))
    print('Pass negate backward!')


def test_log_backward():
    gradient_check(ag.log, ag.Tensor(np.random.rand(5, 4)))
    print('Pass log backward!')


def test_exp_backward():
    gradient_check(ag.exp, ag.Tensor(np.random.randn(5, 4)))
    print('Pass exp backward!')


def test_compute_gradient():
    gradient_check(lambda A,B,C : ag.summation((A@B+C)*(A@B), dim=None),
                   ag.Tensor(np.random.randn(10,9)),
                   ag.Tensor(np.random.randn(9,8)),
                   ag.Tensor(np.random.randn(10,8)), backward=True)
    gradient_check(lambda A,B : ag.summation(ag.broadcast_to(A,shape=(10,9))*B, dim=None),
                   ag.Tensor(np.random.randn(10,1)),
                   ag.Tensor(np.random.randn(10,9)), backward=True)
    gradient_check(lambda A,B,C : ag.summation(ag.reshape(A,shape=(10,10))@B/5+C, dim=None),
                   ag.Tensor(np.random.randn(100)),
                   ag.Tensor(np.random.randn(10,5)),
                   ag.Tensor(np.random.randn(10,5)), backward=True)

    # check gradient of gradient
    x2 = ag.Tensor([6])
    x3 = ag.Tensor([0])
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1
    print('Pass all tests!')
    

if __name__ == '__main__':
    test_add_backward()
    test_add_scaler_backward()
    test_multiply_backward()
    test_mul_scaler_backward()
    test_divide_backward()
    test_divide_scalar_backward()
    test_transpose_backward()
    test_reshape_backward()
    test_broadcast_to_backward()
    test_summation_backward()
    test_matmul_simple_backward()
    test_matmul_batched_backward()
    test_negate_backward()
    test_exp_backward()
    test_compute_gradient()
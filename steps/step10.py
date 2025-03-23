# 테스트

import numpy as np
import unittest

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# 지금까지의 테스트는 기댓값을 직접 계산해 입력했으나, 기울기 확인(gradient checking)이라는 방법으로 자동화할 수 있습니다.
# 수치 미분으로 구한 결과와 역전파로 구한 결과를 비교하는 방법입니다.
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SqureTest(unittest.TestCase):
    # 순전파 테스트
    def test_forward(self):
        x = Variable(np.array(2.0)) # 입력값
        y = square(x)
        expected = np.array(4.0) # 출력값
        self.assertEqual(y.data, expected)
    
    # 역전파 테스트 
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    # 자동 테스트
    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # 무작위 입력값 생성
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad) # 거의 일치하는지 확인
        self.assertTrue(flg)

# 다음 명령어로 실행합니다.
# python -m unittest steps/step10.py
# 혹은 파일 끝에 다음 코드를 추가하면 실행만으로 평가할 수 있습니다.
# unittest.main()
# OK: 테스트 성공

# 테스트 파일들은 하나의 장소에 모아 관리하는 것이 일반적이고, 테스트 코드는 tests 디렉터리에 모아두어 관리할 수 있습니다.
# python -m unittest discover tests

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None 
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self,x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 순전파
x = Variable(np.array(0.5))
y = square(exp(square(x)))
# 역전파
y.backward()
print(x.grad) # 3.297442541400256

unittest.main()
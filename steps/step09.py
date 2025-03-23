# 함수를 더 편리하게

import numpy as np

# 1. 파이썬 함수로 이용하기
# 기존에는 Square 클래스의 인스턴스를 생성한 다음, 그 인스턴스를 호출하는 두 단계로 구분해 진행해야 했습니다.
# 더 편리하게 square, exp 함수를 생성합니다.
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

# 2. backware 메소드 간소화
# Variable 클래스의 backward 메서드에 검증 코드를 추가하여, grad가 None인 경우 자동으로 미분값을 생성하도록 합니다.

# 3. ndarray만 취급
# Variable 클래스의 data는 ndarray 인스턴스만 취급하도록 수정합니다.
def as_array(x):
    if np.isscalar(x): # 입력이 스칼라인 경우 ndarray 인스턴스로 변환
        return np.array(x)
    return x

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
        if self.grad is None: # 변수의 grad가 None이면 자동으로 미분값 생성
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
y = square(exp(square(x))) # 연속하여 적용
# a = square(x)
# b = exp(a)
# y = square(b)

# 역전파
# y.grad = np.array(1.0) 생략 가능
y.backward()
print(x.grad) # 3.297442541400256

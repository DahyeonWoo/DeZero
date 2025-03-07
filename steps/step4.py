# 수치 미분
'''
미분이란: 간단히 변화율을 의미, 정확하게는 '극한으로 짧은 시간(순간)에서 변화량'
y=f(x)가 어떤 구간에서 미분 가능하다면 아래 식은 해당 구간의 '모든 x'에서 성립합니다. f'(x)는 f(x)의 도함수입니다.
f'(x)=lim(h->0){(f(x+h)-f(x))/h}

수치미분(numerical differentiation)이란, 미세한 차이를 이용하여 함수의 변화량을 구하는 방법입니다. 
수치 미분은 극한을 취급할 수 없어 작은 값을 사용하여 '진정한 미분'을 근사합니다.

이 근사 오차를 줄이는 방법으로는 중앙차분(centered difference)이라는 게 있습니다.
중앙차분은 f(x)와 f(x+h)의 차이를 구하는 대신 f(x-h)와 f(x+h)의 차이를 구합니다.
전진 차분(forward difference)는 f(x)와 f(x+h)의 차이를 구하는데, 수치미분 시 전진차분보다 중앙차분이 진정한 미분값에 가깝습니다. (테일러 정리)

다만 수치 미분의 결과에는 오차가 포함되어 있습니다. 대부분 경우 매우 작지만 어떤 계산이냐에 따라 커질 수도 있습니다.
수치 미분의 더 심각한 문제는 계산량이 많다는 점입니다. 변수가 여러 개인 계산을 미분할 경우 변수 각각을 미분해야 하기 때문입니다. 신경망에서는 매개변수를 수백만 개 이상 사용하는 경우가 잦으므로 현실적이지 않습니다.
그래서 등장한 것이 역전파입니다.

덧붙여, 수치 미분은 구현하기 쉽고 거의 정확한 값을 얻을 수 있습니다. 이에 비해 역전파는 복잡한 알고리즘이라 구현하며 버그가 들어가기 쉽습니다.
그래서 역전파를 정확하게 구현했는지 확인하기 위해 수치 미분의 결과를 이용하는데, 이를 기울기 확인(gradient checking)이라고 합니다.
'''
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y) 
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self,x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

# 수치미분 구현
def numerical_diff(f, x, eps=1e-4): # f:미분 대상이 되는 함수, x:미분을 계산하는 변수, eps:작은 값
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f,x)
print(dy) #x^2의 미분, 대략 2x에 가까움 / 4.000000000004

# 합성함수 미분 구현
#  y =(e^x^2)^2이라는 계산에 대한 미분 dy/dx 계산
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
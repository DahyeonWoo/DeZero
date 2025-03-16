# 수동 역전파

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 미분값도 함께 저장

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y) 
        self.input = input # 입력 변수를 기억(보관)한다.
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, x): # 미분을 계산하는 역전파 (backward 메서드)
        raise NotImplementedError()
    
class Square(Function):
    def forward(self,x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy # x^2의 미분(dy/dx)은 2x
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# 순전파 (합성함수)
# x -> A(Square) -> a -> B(Exp) -> b -> C(Square) -> y
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 역전파
# x.grad <- x.backward <- a.grad <- B.backward <- b.grad <- C.backward <- y.grad(=1)
# 일반적으로 역전파는 손실함수에서 시작되며, 이때 y.grad의 입력이 1로 쓰입니다. 
## 손실함수가 1인 이유: 손실함수는 모델의 예측이 얼마나 틀렸는지를 측정하는 함수이고, 현재 손실 함수의 변화율을 자기 자신에 대해 미분한 값으로 미분값은 1입니다.
## 역전파 목적: 역전파의 목적으로는 손실 함수의 값을 최소화하여 모델이 더 나은 예측을 하게 하는 것이 있습니다.즉, 손실 함수를 줄이기 위해 가중치(Weight)를 조정하는 과정입니다.
### 역전파는 기본적으로 오차(손실)가 뉴런을 거슬러 올라가며 전파되는 과정입니다. 이를 통해 신경망의 각 가중치가 출력에 얼마나 영향을 주는지 계산하고, 이를 기반으로 가중치를 조정하여 최적화하는 것이 핵심입니다.
### 역전파의 핵심 개념은 "미분을 활용하여 신경망의 학습을 가능하게 하는 과정"입니다.
#### 순전파 -> 예측값 (y^), 역전파 -> 손실(𝐿)을 기반으로 가중치 조정
# 하지만  현재 함수는 단순 계산 함수이므로, x.grad는 입력 x에 대한 미분값(기울기)일 뿐, 신경망의 학습을 위한 가중치 업데이트 과정은 아닙니다. 
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad) # 3.297442541400256
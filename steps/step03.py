# 함수 연결

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

# 새로운 함수 구현
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
# 함수 연결
# Function의 __call__ 메서드는 입력과 출력이 모두 Variable 인스턴스이므로 자연스레 DeZero 함수들을 연이어 사용할 수 있음

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

# 3개의 함수를 연이어 적용함
# 모두 Vairable 인스턴스이기 때문에 적용 가능
# 이렇듯 여러 함수를 순서대로 적용하여 만들어진 변환 전체를 하나의 큰 함수로 볼 수도 있으며, 이를 함성 함수(composite funciton)이라고 함
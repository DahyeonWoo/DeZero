# function 클래스 구현

# 노드들을 화살표로 연결해 계산 과정을 표혆산 그림을 계산 그래프(computational graph)라고 함
 
# Function 클래스는 Variable 인스턴스를 입력받아 Variable 인스턴스를 출력합니다.
# Variable 인스턴스의 실제 데이터는 인스턴스 변수인 data에 존재

# Function 클래스는 기반 클래스로서, 모든 함수에 공통되는 기능 구현
# 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현

import numpy as np 

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input): # __call__ 메서드는 파이썬의 특수 메서드로 이후 f = Function() 선언 후 f(..) 형태로 메서드를 호출 가능
        x = input.data # 데이터를 꺼낸다
        y = self.forward(x) # 구체적인 계산은 forward에서 한다
        output = Variable(y) # Variable 형태로 되돌린다
        return output
    
    def forward(self, x):
        raise NotImplementedError() # 구체적인 로직은 하위 클래스에서 구현 #상속하여 구현해야 함 명시적 표시
    
class Square(Function):
    def forward(self,x):
        return x ** 2
    
x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y)) # type 함수는 객체의 클래스를 반환
print(y.data) # 10의 제곱인 100 반환
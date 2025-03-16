# 재귀에서 반복문으로

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 재귀 방식입니다.
    # 재귀는 함수를 호출할 때마다 중간 결과를 메모리에 유지하면서 (스택에 쌓으면서) 처리를 이어가며, 일반적으로 반복문 방식의 효율이 더 좋은 이유입니다. 그러나 요즘 컴퓨터는 메모리가 넉넉해서 조금 더 사용하는 것은 크게 문제가 되지 않습니다.
    def backward_old(self):
        f = self.creator # 1. 함수를 가져옵니다.
        if f is not None:
            x = f.input # 2. 함수의 input을 가져옵니다.
            x.grad = f.backward(self.grad) # 3. 함수의 backward 메서드를 호출합니다.
            x.backward() # 하나 앞 변수의 backward 메서드를 호출합니다. (재귀)
    
    # 반복문 방식입니다.
    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수를 가져옵니다.
            x, y = f.input, f.output # 함수의 입력과 출력을 가져옵니다.
            x.grad = f.backward(y.grad) # backward 메서드를 호출합니다.
            
            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가합니다.

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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
y.grad = np.array(1.0)
y.backward()
print(x.grad) # 3.297442541400256

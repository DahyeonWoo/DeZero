# 역전파 자동화

# 일반적인 계산(순전파)을 한 번만 해주면 어떤 계산이라도 상관없이 역전파가 자동으로 이루어지는 구조를 만들 것입니다.
# Define-by-Run의 핵심 내용
## Defin-by-Run: 딥러닝에서 수행하는 계산들을 계산 시점에 '연결'하는 방식으로, '동적 계산 그래프'라고도 합니다. 
# 지금까지의 그래프들은 모두 일직선으로 늘어선 계산이므로, 함수의 순서를 리스트 형태로 저장해두면 나중에 거꾸로 추적하는 식으로 역전파를 자동화할 수 있습니다.
# 그러나 분기가 있는 계산 그래프나 같은 변수가 여러 번 사용되는 복잡한 계산 그래프는 단순히 리스트로 저장하는 식으로는 풀 수 없습니다. 우리 목표는 아무리 복잡한 계산 그래프도 역전파를 자동으로 할 수 있는 구조입니다.
## 사실 Wengent List(tape)를 활용하면 그래프 역전파도 가능하나, 다른 방법을 알아 볼 것입니다.

# 역전파 자동화의 시작
# 함수 입장에서 변수는 '입력'과 '출력'에 쓰입니다.
# 변수 입장에서 함수는 '창조자(creator)'입니다.

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 변수에서 하나 앞의 변수로 거슬러 올라가는, 반복 작업을 자동화할 수 있도록 새로운 메서드 추가합니다. # 자동 미분
    def backward(self):
        f = self.creator # 1. 함수를 가져옵니다.
        if f is not None:
            x = f.input # 2. 함수의 input을 가져옵니다.
            x.grad = f.backward(self.grad) # 3. 함수의 backward 메서드를 호출합니다.
            x.backward() # 하나 앞 변수의 backward 메서드를 호출합니다. (재귀)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 출력 변수에 창조자를 설정합니다.
        self.input = input
        self.output = output # 출력도 저장합니다.
        return output
        # 이렇듯 DeZero의 동적 계산 그래프는(Dynamic Computational Graph)는 실제 계산이 이루어질 때 변수(상자)에 관련 '연결'을 기록하는 방식으로 만들어집니다. 
        # 체이너와 파이토치의 방식도 이와 비슷합니다.
    
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

# 계산 그래프의 노드들을 거꾸로 거슬러 올라가봅니다.
# assert는 단언하다라는 뜻으로, True가 아니면 예외가 발생합니다.
assert y.creator == C
assert y.creator.input == b 
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a 
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# 이러한 '연결'은 실제로 계산을 수행하는 시점에(순전파로 데이터를 흘려보낼 때) 만들어진다는 점입니다.
# 이러한 특성에 이름을 붙인 것이 Define-by-Run입니다. 데이터를 흘려보냄으로써(Run함으로써) 연결이 규정된다는(Define된다는) 뜻입니다.
# 또하느 노드들의 연결로 이루어진 데이터 구조를 linked likst라고 합니다. 노드는 그래프를 구성하는 요소이며, 링크는 다른 노드를 가리키는 참조를 뜻합니다.

# 역전파 도전! (1)
# y부터 b까지의 역전파
# b<-input<-C<-creator<-y
y.grad = np.array(1.0)

C = y.creator # 1. 함수를 가져옵니다.
b = C.input # 2. 함수의 입력을 가져옵니다.
b.grad = C.backward(y.grad) # 3. 함수의 backward 메소드를 호출합니다.

# b에서 a로의 역전파
# a<-input<-B<-creator<-b
B = b.creator # 1. 함수를 가져옵니다.
a = B.input # 2. 함수의 입력을 가져옵니다.
a.grad = B.backward(b.grad) # 3. 함수의 backward 메서드를 호출합니다.

# a에서 x로의 역전파
# x<-input<-A<-creator<-a
A = a.creator # 1. 함수를 가져옵니다.
x = A.input # 2. 함수의 입력을 가져옵니다.
x.grad = A.backward(a.grad) # 3. 함수의 backward 메서드를 호출합니다.

print(x.grad) # 3.297442541400256


# 역전파 도전! (2) # 자동 미분
# x.grad <- x.backward <- a.grad <- B.backward <- b.grad <- C.backward <- y.grad(=1)
# x <- input <- A <- creator <- a <- input <- B <- creator <- b <- input <- C <- creator <- y
y.grad = np.array(1.0)
y.backward()
print(x.grad) # 3.297442541400256

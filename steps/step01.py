# 변수 클래스 구현

import numpy as np 

class Variable:
    def __init__(self, data):
        self.data = data

data = np.array(1.0) # 다차원 배열 생성
x = Variable(data)
print(x.data) # 1.0

x.data = np.array(2.0) # 새로운 데이터 대입
print(x.data) # 2.0

print(x.data.ndim) # 다차원 배열의 차원 수, 0

y = Variable(data)
y.data = np.array([1,2,3])
print(y.data.ndim) # 1

z = Variable(data)
z.data = np.array([[1,2,3],
            [4,5,6]])
print(z.data.ndim) # 2
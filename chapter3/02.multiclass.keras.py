import matplotlib.pyplot as plot
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn.utils import shuffle

# 학습에 필요한 데이터 정리
M = 2
K = 3
n = 100
N = n * K

# 샘플 데이터
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])

Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

# 모델 정의
model = Sequential([
    Dense(input_dim=M, units=K),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

# 미니 배치의 크기
minibatch_size = 50

# 학습
model.fit(X, Y, epochs=20, batch_size=minibatch_size)

# 분류가 잘 되었는지 테스트
X_, Y_ = shuffle(X, Y)
classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10], batch_size=minibatch_size)

print('classified:')
print(np.argmax(model.predict(X_[0:10]), axis=1) == classes)

print('prob:')
print(prob)

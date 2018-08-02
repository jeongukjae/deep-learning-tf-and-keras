import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist

from sklearn.model_selection import train_test_split

# 학습 데이터 다운로드
(X_train, _y_train), (X_test, _y_test) = mnist.load_data()

# 데이터 정리
N = 10000

X_train = X_train[:N * 4 // 5]
_y_train = _y_train[:N * 4 // 5]
X_test = X_test[:N * 1 // 5]
_y_test = _y_test[:N * 1 // 5]

X_train, X_test =\
    np.reshape(X_train, (X_train.shape[0], -1)),\
    np.reshape(X_test, (X_test.shape[0], -1))

y_train = np.zeros((len(_y_train), 10))
y_train[np.arange(len(_y_train)), _y_train] = 1

y_test = np.zeros((len(_y_test), 10))
y_test[np.arange(len(_y_test)), _y_test] = 1

# 모델
n_in = len(X_train[0])
n_hidden = 200
n_out = 10

model = Sequential([
    Dense(n_hidden, input_dim=n_in),
    Activation('sigmoid'),
    Dense(n_out),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

# 학습
epochs = 1000
batch_size = 100

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# 정확도 평가
loss_and_metrics = model.evaluate(X_test, y_test)
print(loss_and_metrics)

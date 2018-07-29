import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

N = 300
X, y = datasets.make_moons(N, noise=0.3)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2)

model = Sequential([
    Dense(3, input_dim=2),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=500, batch_size=20)

loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)

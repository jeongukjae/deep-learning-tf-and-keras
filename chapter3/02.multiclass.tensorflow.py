import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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

# 샘플 데이터 분포
# plt.plot(X1[:, 0], X1[:, 1], 'ro',
#          X2[:, 0], X2[:, 1], 'bs',
#          X3[:, 0], X3[:, 1], 'g^')
# plt.show()

# 모델 정의
W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

# 미니 배치의 크기, 수
batch_size = 50
n_batches = N

# 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(20):
    X_, Y_ = shuffle(X, Y)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })

# 분류가 잘 되었는지 테스트
X_, Y_ = shuffle(X, Y)
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print('classified:')
print(classified)

print('prob:')
print(prob)

# 결과 plot
plot_w = np.transpose(sess.run(W))
plot_b = sess.run(b)

plot_x1 = np.arange(-10, 20)
plot_y1 = -((plot_w[0][0] - plot_w[1][0]) * plot_x1 +
            plot_b[0] - plot_b[1]) / (plot_w[0][1] - plot_w[1][1])

plot_x2 = np.arange(-10, 20)
plot_y2 = -((plot_w[1][0] - plot_w[2][0]) * plot_x2 +
            plot_b[1] - plot_b[2]) / (plot_w[1][1] - plot_w[2][1])

plt.xlim([-5, 15])
plt.ylim([-5, 15])
plt.plot(X1[:, 0], X1[:, 1], 'ro',
         X2[:, 0], X2[:, 1], 'bs',
         X3[:, 0], X3[:, 1], 'g^')
plt.plot(plot_x1, plot_y1, 'c-')
plt.plot(plot_x2, plot_y2, 'c-')

plt.show()

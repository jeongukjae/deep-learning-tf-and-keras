import tensorflow as tf
import numpy as np

# 모델 설정
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

cross_entropy = -tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 훈련 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# 훈련
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(200):
        sess.run(train_step, feed_dict={
            x: X,
            t: Y
        })

    # 결과
    result = correct_prediction.eval(session=sess, feed_dict={
        x: X,
        t: Y
    })
    print("올바르게 분류되었는가?")
    print(result)

    # 결과 2
    result = y.eval(session=sess, feed_dict={
        x: X,
        t: Y
    })
    print("OR 확률은?")
    print(result)

    print('w: ', sess.run(w))
    print('b: ', sess.run(b))

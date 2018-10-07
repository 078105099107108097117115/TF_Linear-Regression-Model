import numpy as np
import tensorflow as tf
x_data = np.random.randn(300,3)
weights = [0.4,0.34,0.12]
bias = -0.1

noise = np.random.randn(1,300) * 0.1
y_data = np.matmul(weights,x_data.T) + bias + noise

NUM_OF_STEPS = 10
g = tf.Graph()
wb_ = []

with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape = None)

    with tf.name_scope('inference layer') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x) + b)

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
    
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_OF_STEPS):
            sess.run(train,{x:x_data, y_true:y_data})
            if(step % 5 == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))
        
        print(10,sess.run(wb_))
        print(10,sess.run([w,b]))



import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
var_grad = tf.gradients(loss, [W,b])

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

for i in range(500):
	print('iteration:',i)
	sess.run(train, {x:x_train, y:y_train})
	# print('loss:', sess.run(loss, {x:x_train,y:y_train}))
	# print('y_pred:', sess.run(linear_model, {x:x_train, y:y_train}))
	# print('y:', y_train)
	print('gradients for weights', sess.run(var_grad, {x:x_train, y:y_train}))

	# evaluate training accuracy
	curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

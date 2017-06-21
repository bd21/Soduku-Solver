import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) #same type as before, implicitly

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)

print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b #convenient to use variable

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2,4]}))
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# so far we've run everything based on constant nodes (no inputs, always produce a type of output)
# now we are going to add variables to the graph

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

#variables are uninitialized until we manually do so

init = tf.global_variables_initializer()
sess.run(init)

# run with multiple values of x

print(sess.run(linear_model, {x:[1,2,3,4]}))

# we don't know how good this linear model is, however.
# so we will write a loss function to calculate how far away
#   current model is from the provided data
# for linear models this is the least squared regression

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # this is the loss value










import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
input > weights > HL1: sum and act.funct. > weights > HL2: sum and act.funct > weights > OUTPUT LAYER: sum & act.funct.

Compare output to intended output with a cost function (example: cross entropy.
Optimization algorithm (aka optimizer: AdamOptimizer, SGD, AdaGrad) to minimise the cost function.

Backpropagation

Feed forward + Backpropagation = Epoch
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# 10 possible values (classes), from 0 to 9.
# One element is hot (one_hot = True), i.e value 1, and the
# other elements are cold (off), i.e value 0.
#0 = [1,0,0,0,0,0,0,0,0,0]
#1 = [0,1,0,0,0,0,0,0,0,0]
#2 = [0,0,1,0,0,0,0,0,0,0]
#3 = [0,0,0,1,0,0,0,0,0,0]

# Number of nodes in each layer.
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

# Number of used images.
batch_size = 100

# height x width = total
# [0, total] means matrix flatten --> one row
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # 784 inputs, 500 nodes in the HL1.
    # Each one of these inputs is connected with each node.
    # Total weights: 784 * n_nodes_hl1
    # output_neuron{j}_in_HL1 = act.funct(SUM{inputs_i * weight_i} + bias_j)
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])

    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])

    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

    #return output
    return tf.nn.relu(output)

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feedforward and backpropagation
    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

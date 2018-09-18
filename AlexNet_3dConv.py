import numpy as np
import tensorflow as tf
from LoadData import loadData
import os

# class describe
groupMap = {'CN': np.array([1, 0, 0]), 'MCI': np.array([0, 1, 0]), 'AD': np.array([0, 0, 1])}
group = {0: 'CN', 1: 'MCI', 2: 'AD'}
num_labels = 3

# file
data_path = '/home/share/data/ADNI1_Annual_2_Yr_3T'
summary_path = '/home/zhengjiaqi/zhengjiaqi/summary/'

# device
device = '/gpu:1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

trainNum = 270
Xshape = (111, 111, 111)
X, Y = loadData(data_path, groupMap, Xshape)
X = X.reshape((-1, 111, 111, 111, 1))
Y = Y.reshape((-1, 3))
X_train = X[:270]
X_test = X[270:]
Y_train = Y[:270]
Y_test = Y[270:]

# Accuracy function
def get_accuracy(predictions, labels):
    return 100 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)), tf.float32))

graph = tf.Graph()
with tf.device(device):
    with graph.as_default():

        predict = tf.Variable(False)
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, 111, 111, 111, 1))
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    
        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
          [9, 9, 9, 1, 96], stddev=0.1))
    
        layer1_biases = tf.Variable(tf.zeros([96]))
    
        layer2_weights = tf.Variable(tf.truncated_normal(
          [5, 5, 5, 96, 256], stddev=0.1))
         
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[256]))
    
        layer3_weights = tf.Variable(tf.truncated_normal(
          [3, 3, 3, 256, 384], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[384]))
    
        layer4_weights = tf.Variable(tf.truncated_normal(
          [3, 3, 3, 384, 384], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[384]))
    
        layer5_weights = tf.Variable(tf.truncated_normal(
          [3, 3, 3, 384, 256], stddev=0.1))
        layer5_biases = tf.Variable(tf.constant(1.0, shape=[256]))
    
        layer6_weights = tf.Variable(tf.truncated_normal(
          [7 * 7 * 7 * 256, 4096], stddev=0.1))
        layer6_biases = tf.Variable(tf.constant(1.0, shape=[4096]))
    
        layer7_weights = tf.Variable(tf.truncated_normal(
          [4096, 4096], stddev=0.1))
        layer7_biases = tf.Variable(tf.constant(1.0, shape=[4096]))
    
        layer8_weights = tf.Variable(tf.truncated_normal(
          [4096, num_labels], stddev=0.1))
        layer8_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    
    
        # MODEL
        def model(data):
            # Conv1         
            conv1 = tf.nn.conv3d(data, layer1_weights, [1, 4, 4, 4, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + layer1_biases)
    
            #Pool1
            pool1 = tf.nn.max_pool3d(hidden1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    
            # Conv2
            conv2 = tf.nn.conv3d(pool1, layer2_weights, [1, 1, 1, 1, 1],padding='SAME')
            hidden2 = tf.nn.relu(conv2 + layer2_biases)
    
            # Conv3
            conv3 = tf.nn.conv3d(hidden2, layer3_weights, [1, 1, 1, 1, 1],padding='SAME')
    
            # Conv4
            conv4 = tf.nn.conv3d(conv3, layer4_weights, [1, 1, 1, 1, 1], padding='SAME')
    
            # Conv5
            conv5 = tf.nn.conv3d(conv4, layer5_weights, [1, 1, 1, 1, 1], padding='SAME')
    
            #Pool2
            pool2 = tf.nn.max_pool3d(conv5, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
            
            normalize3_flat = tf.reshape(pool2, [-1, 7 * 7 * 7 * 256])
    
            #FC1
            fc1 = tf.tanh(tf.add(tf.matmul(normalize3_flat, layer6_weights), layer6_biases))
            dropout1 = tf.nn.dropout(fc1, 0.5)
    
            #FC2
            fc2 = tf.tanh(tf.add(tf.matmul(dropout1, layer7_weights), layer7_biases))
            dropout2 = tf.nn.dropout(fc2, 0.5)
    
            #FC3
            res = tf.nn.softmax(tf.add(tf.matmul(dropout2, layer8_weights), layer8_biases))
            return res
    
        
         # Training computation
        local_res = model(tf_train_dataset)
    
        with tf.name_scope("cost_function") as scope:
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_train_labels * tf.log(local_res), reduction_indices=[1]))
            tf.summary.scalar("cost_function", cross_entropy)
        
        # Optimizer
        train_step = tf.train.MomentumOptimizer(0.00014, 0.5).minimize(cross_entropy)
    
        # Predictions for the training, validation, and test data
        with tf.name_scope("accuracy"):
            accuracy = get_accuracy(local_res, tf_train_labels)
            tf.summary.scalar("accuracy", accuracy)
        
    
        valid_prediction = tf.nn.softmax(model(tf_train_dataset))
        print('Graph was built')
        
        merged_summary_op = tf.summary.merge_all()


# Graph
batch_size = 10

# Session
epochs = 30
steps_per_epoch = int(Y_train.shape[0]/batch_size)
print('STEPS %d' % steps_per_epoch)

with tf.Session(graph=graph) as session:
    train_summary = tf.summary.FileWriter(summary_path, session.graph)
    session.run(tf.global_variables_initializer())
    
    for epch in range(0, epochs):
        print('EPOCH %d' % epch)

        for step in range(steps_per_epoch):
            offset = step * batch_size

            # Generate a minibatch.
            batch_data = X_train[offset:(offset + batch_size)].astype('float32')
            batch_labels = Y_train[offset:(offset + batch_size)]

            train_step.run(feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()


            summary_str, _ =  session.run([merged_summary_op, train_step],
                                   feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels},
                                   options=run_options,
                                   run_metadata=run_metadata)

            train_summary.add_run_metadata(run_metadata, 'step%03d' % (int(step)+(steps_per_epoch * (epch+1))))
            train_summary.add_summary(summary_str, step)

            train_accuracy, train_cross_entropy = session.run([accuracy, cross_entropy], feed_dict={
                tf_train_dataset:batch_data, tf_train_labels: batch_labels})

            print("Step %d" % step, "Minibatch accuracy: %.1f%%" % train_accuracy,
                  "Cross entropy: %.1f" % train_cross_entropy)

    test_accuracy = session.run(accuracy, feed_dict={
        tf_train_dataset: X_test, tf_train_labels: Y_test})
    print("Test accuracy: %.1f%%" % test_accuracy)
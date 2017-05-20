import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import	lenet5_infernece
import lenet5_train
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def evaluate(X_test,y_test_lable):
	with tf.Graph().as_default() as g:
	
		# 定義輸出為4維矩陣的placeholder
		x_ = tf.placeholder(tf.float32, [None, lenet5_train.INPUT_NODE],name='x-input')	
		x = tf.reshape(x_, shape=[-1, 28, 28, 1])
	
		y_ = tf.placeholder(tf.float32, [None, lenet5_train.OUTPUT_NODE], name='y-input')
	
		regularizer = tf.contrib.layers.l2_regularizer(lenet5_train.REGULARIZATION_RATE)
		y = lenet5_infernece.inference(x,False,regularizer)
		global_step = tf.Variable(0, trainable=False)

		# Evaluate model
		pred_max=tf.argmax(y,1)
		y_max=tf.argmax(y_,1)
		correct_pred = tf.equal(pred_max,y_max)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
		test_batch_len =int( X_test.shape[0]/lenet5_train.BATCH_SIZE)
		test_acc=[]
		

		test_xs = np.reshape(X_test, (
					X_test.shape[0],
					lenet5_train.IMAGE_SIZE,
					lenet5_train.IMAGE_SIZE,
					lenet5_train.NUM_CHANNELS))
		
		batchsize=lenet5_train.BATCH_SIZE
	
		# 'Saver' op to save and restore all the variables
		saver = tf.train.Saver()
		with tf.Session() as sess:
			
			saver.restore(sess,"./lenet5/lenet5_model")

			for i in range(test_batch_len):
				temp_acc= sess.run(accuracy, feed_dict={x: test_xs[batchsize*i:batchsize*i+batchsize], y_: y_test_lable[batchsize*i:batchsize*i+batchsize]})
				test_acc.append(temp_acc)
				print ("Test  batch ",i,":Testing Accuracy:",temp_acc)	

			t_acc=tf.reduce_mean(tf.cast(test_acc, tf.float32))	
			print("Average Testing Accuracy=",sess.run(t_acc))
			return

def main(argv=None):
	#### Loading the data
	X_train, y_train = lenet5_train.load_mnist('..\mnist', kind='train')
	print('X_train Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1])) #X_train=60000x784
	X_test, y_test = lenet5_train.load_mnist('mnist', kind='t10k')					 #X_test=10000x784
	print('X_test Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
	mms=MinMaxScaler()
	X_train=mms.fit_transform(X_train)
	X_test=mms.transform(X_test)

	y_train_lable = lenet5_train.encode_labels(y_train,10)
	y_test_lable = lenet5_train.encode_labels(y_test,10)
	##============================
	
	evaluate(X_test,y_test_lable)

if __name__ == '__main__':
	main()

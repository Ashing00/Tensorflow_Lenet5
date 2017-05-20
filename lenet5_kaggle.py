import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import	lenet5_infernece
import lenet5_train
import os,csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def nomalizing(array):	
	m,n=np.shape(array)	 
	for i in range(m):	 
		for j in range(n):	 
			if array[i,j]!=0:  
				array[i,j]=1  
	return array  
	
def toInt(array):  
	array=np.mat(array)	 
	m,n=np.shape(array)	 
	newArray=np.zeros((m,n))  
	for i in range(m):	 
		for j in range(n):	 
				newArray[i,j]=int(array[i,j])  
	return newArray
def loadTrainData():  
	l=[]  
	with open('train.csv') as file:	 
		 lines=csv.reader(file)	 
		 for line in lines:	 
			 l.append(line) #42001*785	
	l.remove(l[0])	
	l=np.array(l)	
	label=l[:,0]  
	data=l[:,1:]  
	return toInt(data),toInt(label) 
	#return nomalizing(toInt(data)),toInt(label)  
	
def loadTestData():	 
	l=[]  
	with open('test.csv') as file:	
		 lines=csv.reader(file)	 
		 for line in lines:	 
			 l.append(line) #28001*784	 
	l.remove(l[0])	
	data=np.array(l)  
	return toInt(data)
	#return nomalizing(toInt(data))	  
	
def loadTestResult():  
	l=[]  
	with open('knn_benchmark.csv') as file:	 
		 lines=csv.reader(file)	 
		 for line in lines:	 
			 l.append(line)	 
	 #28001*2  
	l.remove(l[0])	
	label=np.array(l)	
	return toInt(label[:,1])	

def saveResult(result):
	with open ('result.csv', mode='w',newline="\n") as write_file:
		writer = csv.writer(write_file)
		writer.writerow(["ImageId","Label"])
		for i in range(len(result)):
			writer.writerow([i+1,result[i]])
		
def saveweight(w1,w2):
	with open ('weight1.csv', mode='w',newline="\n") as write_file:
		writer = csv.writer(write_file)
		for i in range(len(w1)):
			writer.writerow([w1[i]])
	with open ('weight2.csv', mode='w',newline="\n") as write_file2:
		writer = csv.writer(write_file2)
		for i in range(len(w2)):
			writer.writerow([w2[i]])		
			

def evaluate(X_test):
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
		
		kaggle_pred=np.array([])
		
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
				pred_result=sess.run(pred_max, feed_dict={x: test_xs[batchsize*i:batchsize*i+batchsize]})
				kaggle_pred=np.append(kaggle_pred,pred_result)
				kaggle_pred=kaggle_pred.astype(int)
				kaggle_pred=kaggle_pred.tolist()
			
			print("pred_result.length:",len(kaggle_pred))
			#print("pred_result=",kaggle_pred)
			print("Save prediction result...")
			saveResult(kaggle_pred)	
			return

def main(argv=None):
##load kaggle data+	
	print("Load kaggle Mnist data...")
	X_test=loadTestData()	
	print("test_data.shape=",X_test.shape)
	
##load data-
	##============================
	evaluate(X_test)

if __name__ == '__main__':
	main()

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_infernece
import lenet5_train
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
from matplotlib import pyplot as plt

img_num=[0]*20

def evaluate(X_test,y_test_lable,My_Yd):
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
		batchsize=20
		test_batch_len =int( X_test.shape[0]/batchsize)
		test_acc=[]
		
		test_xs = np.reshape(X_test, (
					X_test.shape[0],
					lenet5_train.IMAGE_SIZE,
					lenet5_train.IMAGE_SIZE,
					lenet5_train.NUM_CHANNELS))
		
		# 'Saver' op to save and restore all the variables
		saver = tf.train.Saver()
		#saver = tf.train.import_meta_graph("./mnist/mnist_model.meta")
		with tf.Session() as sess:
			
			saver.restore(sess,"./lenet5/lenet5_model")

			My_test_pred=sess.run(pred_max, feed_dict={x: test_xs[:20]})
			print("期望值：",My_Yd)
			print("預測值：",My_test_pred)
			My_acc = sess.run(accuracy, feed_dict={x: test_xs, y_: y_test_lable})
			print('Test accuracy: %.2f%%' % (My_acc * 100))
			display_result(My_test_pred)		
			return
			
def display_result(my_prediction):	
	img_res=[0]*20
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i in range(20):	 
		img_res[i] = np.zeros((64,64,3), np.uint8)
		img_res[i][:,:]=[255,255,255]
		if (my_prediction[i]%10)==(i%10):
			cv2.putText(img_res[i],str(my_prediction[i]),(15,52), font, 2,(0,255,0),3,cv2.LINE_AA)
		else:
			cv2.putText(img_res[i],str(my_prediction[i]),(15,52), font, 2,(255,0,0),3,cv2.LINE_AA)

	Input_Numer_name = ['Input 0', 'Input 1','Input 2', 'Input 3','Input 4',\
					'Input 5','Input 6', 'Input 7','Input8', 'Input9',\
					'Input 0', 'Input 1','Input 2', 'Input 3','Input 4',\
					'Input 5','Input 6', 'Input 7','Input8', 'Input9',
					]
					
	predict_Numer_name =['predict 0', 'predict 1','predict 2', 'predict 3','predict 4', \
					'predict 5','predict6 ', 'predict 7','predict 8', 'predict 9',\
					'predict 0', 'predict 1','predict 2', 'predict 3','predict 4', \
					'predict 5','predict6 ', 'predict 7','predict 8', 'predict 9',
					]
				
	for i in range(20):
		if i<10:
			plt.subplot(4,10,i+1),plt.imshow(img_num[i],cmap = 'gray')
			plt.title(Input_Numer_name[i]), plt.xticks([]), plt.yticks([])
			plt.subplot(4,10,i+11),plt.imshow(img_res[i],cmap = 'gray')
			plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])
		else:
			plt.subplot(4,10,i+11),plt.imshow(img_num[i],cmap = 'gray')
			plt.title(Input_Numer_name[i]), plt.xticks([]), plt.yticks([])
			plt.subplot(4,10,i+21),plt.imshow(img_res[i],cmap = 'gray')
			plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])
		
	plt.show()			
			
			
def main(argv=None):
	#### Loading the data
	#自己手寫的20個數字
	My_X =np.zeros((20,784), dtype=int) 
	#自己手寫的20個數字對應的期望數字
	My_Yd =np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9], dtype=int) 

	#輸入20個手寫數字圖檔28x28=784 pixel，
	Input_Numer=[0]*20
	Input_Numer[0]="0_5.jpg"
	Input_Numer[1]="1_5.jpg"
	Input_Numer[2]="2_5.jpg"
	Input_Numer[3]="3_5.jpg"
	Input_Numer[4]="4_5.jpg"
	Input_Numer[5]="5_5.jpg"
	Input_Numer[6]="6_5.jpg"
	Input_Numer[7]="7_5.jpg"
	Input_Numer[8]="8_5.jpg"
	Input_Numer[9]="9_5.jpg"
	Input_Numer[10]="0_7.jpg"
	Input_Numer[11]="1_7.jpg"
	Input_Numer[12]="2_7.jpg"
	Input_Numer[13]="3_7.jpg"
	Input_Numer[14]="4_7.jpg"
	Input_Numer[15]="5_7.jpg"
	Input_Numer[16]="6_7.jpg"
	Input_Numer[17]="7_7.jpg"
	Input_Numer[18]="8_7.jpg"
	Input_Numer[19]="9_7.jpg"
	mms=MinMaxScaler()
	for i in range(20):	 #read 20 digits picture
		img = cv2.imread(Input_Numer[i],0)	  #Gray
		img_num[i]=img.copy()
		img=img.reshape(My_X.shape[1])
		My_X[i] =img.copy()

	My_test=mms.fit_transform(My_X)
	My_label_ohe = lenet5_train.encode_labels(My_Yd,10)
	##============================
	
	evaluate(My_test,My_label_ohe,My_Yd)

if __name__ == '__main__':
	main()

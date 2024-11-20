#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import r2_score 
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


# In[2]:


dir_ = 'Outputs'
os.mkdir(dir_)


# In[3]:


# Read the Dataset

path = os.getcwd()
sepsis_files = glob.glob(os.path.join(path, "sepsis/*.csv"))
no_sepsis_files = glob.glob(os.path.join(path, "no_sepsis/*.csv"))


# In[4]:


i = 0
for file in sepsis_files:
    person = pd.read_csv(file)
    if(i==0):
        df = person.mean(axis=0, skipna=True, level=None).to_frame().T
        i = 1
    else:
        df = pd.concat([df, person.mean(axis=0, skipna=True, level=None).to_frame().T], ignore_index=True)
df = df.drop(df.index[500:1000])
for file in no_sepsis_files:
    person = pd.read_csv(file)
    df = pd.concat([df, person.mean(axis=0, skipna=True, level=None).to_frame().T], ignore_index=True)
df = df.drop(df.index[1000:1500])


# In[5]:


# Shuffle the Dataset

df = df.sample(frac=1, ignore_index=True) 


# In[6]:


# Switch Missing Values in the Dataset with Means

msno.matrix(df)


# In[7]:


for column in ['map', 'spo2', 'fio2', 'bilirubin', 'lactate', 'ph', 'pco2', 'po2', 'bicarbonate', 'hemoglobin', 'chloride']:
    mean_value= np.nanmean(df[column])
    df[column] = df[column].fillna(mean_value)


# In[8]:


# Determine Input-Output Arrays

X = df.drop(columns=['fio2', 'ph', 'pco2', 'po2', 'bp_systolic', 'bp_diastolic', 'resp', 'temp', 'spo2', 'wbc', 'bun', 'bicarbonate', 'hemoglobin', 'hematocrit', 'potassium', 'chloride', 'age', 'sirs', 'qsofa', 'sepsis_icd']).to_numpy()
y = df["sepsis_icd"].to_numpy()

# The Scaler Object (Model)

scaler = StandardScaler()

# Fit and Transform the Data

scaled_data = scaler.fit_transform(X)
X = scaled_data

# 5-Fold Cross Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# In[9]:


# R-square Values Between Features

features = ['Heart Rate', 'MAP', 'Bilirubin', 'Creatinine', 'Lactate', 'Platelets', 'GCS']

for first in range(X.shape[1]):
    for second in range(X.shape[1]):
        R_square = r2_score(X[first], X[second]) 
        print('Coefficient of Determination Between', features[first], 'and', features[second], ':   ', R_square) 


# In[10]:


print(X_test)
print(y_test)


# In[11]:


print('X_train.shape:  ', X_train.shape)
print('y_train.shape:  ', y_train.shape)
unique, counts = np.unique(y_train, return_counts=True)
print("Train:  ", dict(zip(unique, counts)))

print('X_test.shape:  ', X_test.shape)
print('y_test.shape:  ', y_test.shape)
unique, counts = np.unique(y_test, return_counts=True)
print("Test:  ", dict(zip(unique, counts)))


# In[12]:


# Settings

n = X_train.shape[1] # number of input features
m = 2*n # number of fuzzy rules

learning_rate = 0.01
epochs = 1000


# In[13]:


# Train

X_train_t = tf.placeholder(tf.float32, shape=[None, n]) # Train the Input
y_train_t = tf.placeholder(tf.float32, shape=None)  # Train the Output

mu = tf.get_variable(name="mu", shape=[m * n], initializer=tf.random_normal_initializer(0, 1))  # Mean of Gaussian MFS
sigma = tf.get_variable(name="sigma", shape = [m * n], initializer=tf.random_normal_initializer(0, 1))  # std_dev of Gaussian MFS
w = tf.get_variable(name="w", shape= [1, m], initializer=tf.random_normal_initializer(0, 1))

rula = tf.reduce_prod(tf.reshape(tf.exp( -0.5* ((tf.tile(X_train_t, (1, m))- mu)**2) / (sigma**2)), (-1, m, n)), axis=2)  # Activations
Y_train_t = tf.reduce_sum(rula*w,axis=1) / tf.clip_by_value(tf.reduce_sum(rula,axis=1), 1e-8, 1e8)


#loss = tf.losses.log_loss(y_train, Y_train) 
loss = tf.losses.sigmoid_cross_entropy(y_train_t, Y_train_t)  # Loss Function
#loss = tf.sqrt(tf.losses.mean_squared_error(y_train, Y_train))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Optimizer
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# In[14]:


# Test

X_test_t = tf.placeholder(tf.float32, shape=[None, n]) # Test the Input
y_test_t = tf.placeholder(tf.float32, shape=None)  # Test the Output

rula_test = tf.reduce_prod(tf.reshape(tf.exp( -0.5* ((tf.tile(X_test_t, (1, m))- mu)**2) / (sigma**2)), (-1, m, n)), axis=2)  # Rule Activation
Y_test_t = tf.reduce_sum(rula_test*w,axis=1) / tf.clip_by_value(tf.reduce_sum(rula_test,axis=1), 1e-8, 1e8)

loss_test = tf.losses.sigmoid_cross_entropy(y_test_t, Y_test_t)  # Loss Function


# In[16]:


# session

x_axis = []
tr_loss, te_loss = [],[]
tr_acc, te_acc = [], []
tr_f1, te_f1 = [], []
tr_prec, te_prec = [], []
tr_rec, te_rec = [], []
init=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
	sess.run(init)

	for e in range(epochs):
		Y_train, loss_tr, _ = sess.run([Y_train_t, loss, optimizer], feed_dict={X_train_t: X_train, y_train_t: y_train})
		Y_test, loss_te    = sess.run([Y_test_t, loss_test], feed_dict={X_test_t: X_test, y_test_t: y_test})
	    
		if (e+1) % 10 == 0:
			x_axis.append(e+1)

			tr_loss.append(loss_tr)
			te_loss.append(loss_te)

			Y_train = np.where(Y_train > 0, 1, 0)
			Y_test = np.where(Y_test > 0, 1, 0)

			acc_tr = accuracy_score(y_train,Y_train)
			acc_te = accuracy_score(y_test,Y_test)

			f1_tr = f1_score(y_train,Y_train)
			f1_te = f1_score(y_test,Y_test)

			prec_tr = precision_score(y_train,Y_train)
			prec_te = precision_score(y_test,Y_test)

			rec_tr = recall_score(y_train,Y_train)
			rec_te = recall_score(y_test,Y_test)

			tr_acc.append(acc_tr)
			te_acc.append(acc_te)
			tr_f1.append(f1_tr)
			te_f1.append(f1_te)	
			tr_prec.append(prec_tr)
			te_prec.append(prec_te)
			tr_rec.append(rec_tr)
			te_rec.append(rec_te)

		if (e+1) % 200 == 0:
			print("Epoch ",e+1,">>>>>>>>>>>>>")
			#print(Y_test) ### debug print
			print("loss      >>>","test:", loss_te,"\t\t train:",loss_tr)
			print("accuracy  >>>","test:", acc_te,"\t train:",acc_tr)
			print("f1-score  >>>","test:", f1_te,"\t train:",f1_tr)
			print("precision >>>","test:", prec_te,"\t train:",prec_tr)
			print("recall    >>>","test:", rec_te,"\t train:",rec_tr)
			print()

			cm_test = confusion_matrix(y_test, Y_test)
			cm_train = confusion_matrix(y_train, Y_train)
	
			plt.figure(int(str(e+1)+'1'))
			disp = ConfusionMatrixDisplay(confusion_matrix=cm_test,display_labels=["Not Sepsis", "Sepsis"])
			disp = disp.plot()
			#plt.show()
			plt.savefig(dir_+"/cf_test-epoch"+str(e+1)+".png",transparent=True)

			plt.figure(int(str(e+1)+'2'))
			disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,display_labels=["Not Sepsis", "Sepsis"])
			disp = disp.plot()
			#plt.show()
			plt.savefig(dir_+"/cf_train-epoch"+str(e+1)+".png",transparent=True)


	# plot accuracy
	plt.figure(1)
	plt.plot(x_axis,tr_acc,label="Train")
	plt.plot(x_axis,te_acc,label="Test")
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Accuracy over Epochs')
	plt.legend()
	#plt.show()
	plt.savefig(dir_+"/acc.png",transparent=True)

	plt.figure(2)
	plt.plot(x_axis,tr_f1,label="Train")
	plt.plot(x_axis,te_f1,label="Test")
	plt.xlabel('Epochs')
	plt.ylabel('F1-score')
	plt.title('F1-score over Epochs')
	plt.legend()
	#plt.show()
	plt.savefig(dir_+"/f1.png",transparent=True)

	plt.figure(3)
	plt.plot(x_axis,tr_prec,label="Train")
	plt.plot(x_axis,te_prec,label="Test")
	plt.xlabel('Epochs')
	plt.ylabel('Precision')
	plt.title('Precision over Epochs')
	plt.legend()
	#plt.show()
	plt.savefig(dir_+"/precision.png",transparent=True)

	plt.figure(4)
	plt.plot(x_axis,tr_rec,label="Train")
	plt.plot(x_axis,te_rec,label="Test")
	plt.xlabel('Epochs')
	plt.ylabel('Recall')
	plt.title('Recall over Epochs')
	plt.legend()
	#plt.show()
	plt.savefig(dir_+"/recall.png",transparent=True)

	plt.figure(5)
	plt.plot(x_axis,tr_loss,label="Train")
	plt.plot(x_axis,te_loss,label="Test")
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss over Epochs')
	plt.legend()
	#plt.show()
	plt.savefig(dir_+"/loss.png",transparent=True)

	mu_fin = sess.run(mu)
	mu_fin = np.reshape(mu_fin, (m, n))
	sigma_fin = sess.run(sigma)
	sigma_fin = np.reshape(sigma_fin, (m,n))
	w_fin = sess.run(w)
	x_axis_mf = np.linspace(-4, 4, 1000)
	for r in range(m):
		plt.figure(r+6)
		plt.title("Rule %d, MF for each feature [ %f ]" % ((r + 1), w_fin[0, r]))
		for i in range(n):
			plt.plot(x_axis_mf, np.exp(-0.5 * ((x_axis_mf - mu_fin[r, i]) ** 2) / (sigma_fin[r, i] ** 2)))
		#plt.show()
		plt.savefig(dir_+"/rule-"+str(r+1)+".png",transparent=True)  


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[196]:


import pandas as pd
import numpy as  np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
# Import `Sequential` from `keras.models`
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


# # Read CSV files - trojan.


dir = "~/Data/"

tj_bench = ["AES100", "AES200", "AES300", "AES400", "AES500", "AES600", "AES700", "AES800", "AES900", "AES1000"]

tj_bench_3_column = ["AES300", "AES500", "AES600"]
result = []

for i in tj_bench:
	print("**************************************************************")
	print("Reading Trojan :", i)
	print("**************************************************************")
	df_trojan_high_freq = pd.read_csv(f"{dir}/{i}/data_0.000_00_1.csv", names=['Input', 'Expected Output', 'Real Output', 'Additional Output'])
	df_trojan_low_freq = pd.read_csv(f"{dir}/{i}/data_0.000_00_2.csv", names=['Input', 'Expected Output', 'Real Output', 'Additional Output'])

	# # Add value to the Additional Output columns:

	# In[157]:
	if i in tj_bench_3_column:
		df_trojan_high_freq['Additional Output'] = '0000000000000000000000000000000011111111111111110000000011111111'
		df_trojan_low_freq['Additional Output'] =  '0000000000000000000000000000000011111111111111110000000011111111'
													
	if i == "AES100":
		df_trojan_high_freq['Additional Output'] = '0000000000000000111111111111111100000000000000000000000011111111'
		
	# In[158]:


	df_trojan_low_freq.shape


	# # Read CSV files - trojanfree

	# In[159]:


	df_trojanFree_high_freq = pd.read_csv("~/projects/HW_Trojan_ML/TrojanFree/data_0.000_00_0.csv", names=['Input', 'Expected Output', 'Real Output'])
	df_trojanFree_low_freq = pd.read_csv("~/projects/HW_Trojan_ML/TrojanFree/data_0.000_00_1.csv", names=['Input', 'Expected Output', 'Real Output'])


	df_trojanFree_high_freq['Additional Output'] = '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
	df_trojanFree_low_freq['Additional Output'] = '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'

	


	df_trojan_high_freq = df_trojan_high_freq.drop([0,1,2])
	df_trojan_high_freq = df_trojan_high_freq.reset_index(drop=True)
	df_trojan_high_freq.head()


	# # trojan - low-freq

	# In[161]:


	df_trojan_low_freq = df_trojan_low_freq.drop([0,1,2])
	df_trojan_low_freq = df_trojan_low_freq.reset_index(drop=True)
	df_trojan_low_freq.head()


	# # trojanFree - high-freq

	# In[162]:


	df_trojanFree_high_freq = df_trojanFree_high_freq.drop([0,1,2])
	df_trojanFree_high_freq = df_trojanFree_high_freq.reset_index(drop=True)
	df_trojanFree_high_freq.head()


	# # trojanFree - low-freq

	# In[163]:


	df_trojanFree_low_freq = df_trojanFree_low_freq.drop([0,1,2])
	df_trojanFree_low_freq = df_trojanFree_low_freq.reset_index(drop=True)
	df_trojanFree_low_freq.head()


	# In[164]:


	def binaryToDecimal(n): 
	    return int(n,2)

	def binaryToHex(n): 
	    bin_to_dec = int(n,2)
	    bin_to_hex = hex(bin_to_dec)
	    return bin_to_hex


	# # Convert binary to Decimal

	# In[165]:


	# trojan - high-freq

	df_trojan_high_freq_dec = df_trojan_high_freq.applymap(binaryToDecimal)
	df_trojan_high_freq_dec.head()


	# In[166]:


	# trojan - low-freq
	df_trojan_low_freq_dec = df_trojan_low_freq.applymap(binaryToDecimal)
	df_trojan_low_freq_dec.head()


	# In[167]:


	# trojanFree - high-freq
	df_trojanFree_high_freq_dec = df_trojanFree_high_freq.applymap(binaryToDecimal)
	df_trojanFree_high_freq_dec.head()


	# In[168]:


	# trojanFree - low-freq
	df_trojanFree_low_freq_dec = df_trojanFree_low_freq.applymap(binaryToDecimal)
	df_trojanFree_low_freq_dec.head()


	# In[169]:


	# trojan - high-freq
	freq_high = 1/.17

	df_trojan_high_freq_dec['Frequency'] = freq_high
	df_trojan_high_freq_dec['Labels'] = 1
	df_trojan_high_freq_dec.head()
	df_trojan_high_freq_dec.shape


	# In[170]:


	# trojan - low-freq
	freq_low = 1/5
	df_trojan_low_freq_dec['Frequency'] = freq_low
	df_trojan_low_freq_dec['Labels'] = 1
	df_trojan_low_freq_dec.head()
	# df_trojan_low_freq_dec.shape


	# In[171]:


	# trojanFree - high-freq
	freq_high = 1/.17

	df_trojanFree_high_freq_dec['Frequency'] = freq_high
	df_trojanFree_high_freq_dec['Labels'] = 0
	df_trojanFree_high_freq_dec.head()
	# df_trojanFree_high_freq_dec.shape


	# In[172]:


	# trojanFree - low-freq
	freq_low = 1/5
	df_trojanFree_low_freq_dec['Frequency'] = freq_low
	df_trojanFree_low_freq_dec['Labels'] = 0
	df_trojanFree_low_freq_dec.head()
	# df_trojanFree_low_freq_dec.shape


	# # Concatinate dataframes

	# In[173]:


	# trojan
	frames_trojan = [df_trojan_high_freq_dec, df_trojan_low_freq_dec]
	df_trojan = pd.concat(frames_trojan)
	df_trojan.head()


	# In[174]:


	# trojanFree
	frames_trojanFree = [df_trojanFree_high_freq_dec, df_trojanFree_low_freq_dec]
	df_trojanFree = pd.concat(frames_trojanFree)
	df_trojanFree.head()


	# # Concatinate trojan and trojanFree dataframes

	# In[178]:


	frames_all = [df_trojan, df_trojanFree]
	df = pd.concat(frames_all, ignore_index = True)
	print(df.head())

	# Normalize dataset

	df[['Input', 'Expected Output', 'Real Output', 'Additional Output']] = df[['Input', 'Expected Output', 'Real Output', 'Additional Output']]/3.40e+38
	print(df.head())


	# # shuffle the dataset

	# In[187]:


	df = df.sample(frac=1).reset_index(drop=True)
	# print(df.head())
	# df.tail()


	# In[179]:


	print(df_trojan_high_freq_dec.shape)
	print(df_trojan_low_freq_dec.shape)
	print(df_trojan.shape)
	print(df_trojanFree_high_freq_dec.shape)
	print(df_trojanFree_low_freq_dec.shape)
	print(df_trojanFree.shape)
	print(df.shape)


	# # Split the dataset into 70-30

	# In[192]:


	# Specify the data 
	y = df['Labels']

	# Specify the target labels
	X = df.drop(['Labels'], axis = 1)

	# Split the data up in train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


	print("y_train", y_train.shape)
	print("y_test", y_test.shape)

	# # Standardize The Data

	# In[194]:


	# Import `StandardScaler` from `sklearn.preprocessing`
	# from sklearn.preprocessing import StandardScaler

	# Define the scaler 
	scaler = StandardScaler().fit(X_train)

	# Scale the train set
	X_train = scaler.transform(X_train)

	# Scale the test set
	X_test = scaler.transform(X_test)


	# # 2. Define Keras Model

	# In[195]:


	# Initialize the constructor
	model = Sequential()

	# Add an input layer 
	model.add(Dense(8,  input_dim=5, activation='relu'))

	# Add one hidden layer 
	model.add(Dense(6, activation='relu'))

	# Add one more hidden layer 
	model.add(Dense(5, activation='relu'))

	# Add one more hidden layer 
	# model.add(Dense(3, activation='relu'))

	# Add an output layer 
	model.add(Dense(1, activation='sigmoid'))


	# # 3. Compile Keras Model

	# In[ ]:


	# compile the keras model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


	# # 4. Fit Keras Model

	# In[ ]:


	# fit the keras model on the dataset
	model.fit(X_train, y_train, epochs=5, batch_size=500, verbose=1)


	# # 5. Evaluate Keras Model

	# In[ ]:


	# evaluate the keras model
	_, accuracy = model.evaluate(X_test, y_test,verbose=1)
	print('Accuracy: ', accuracy*100)


	# # Make Predictions

	# In[ ]:

	# make probability predictions with the model

	predictions = model.predict(X_test)

	# round predictions 
	y_pred = [int(round(x[0])) for x in predictions]

	# In[ ]:


	# show prediction result:
	print("Prediction for first 10 test sample: ")
	print("prediction \n", y_pred[:10])
	print("actual label ", y_test[:10])

	# Import the modules from `sklearn.metrics`
	from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

	# Confusion matrix
	print("Confusion matrix", confusion_matrix(y_test, y_pred))
	# Precision 

	print("Precision", precision_score(y_test, y_pred))
	precision = precision_score(y_test, y_pred)
	# Recall
	print("Recall", recall_score(y_test, y_pred))
	recall = recall_score(y_test, y_pred)

	# F1 score
	print("F1 score", f1_score(y_test,y_pred))
	f1 = f1_score(y_test,y_pred)

	result.append({i : f"Accuracy: {round(accuracy*100,2)}, Precision: {round(precision,2)}, Recall: {round(recall,2)}, F1_score: {round(f1,2)}"}) 

print(result)

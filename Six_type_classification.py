# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:14:59 2018

@author: y.wei
"""
# Normalize time series data
#import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
np.random.seed(7)
#data = np.loadtxt('type_111_202_131_204.txt')
data_1 = np.genfromtxt('type_200_202_131_113.txt', delimiter=',', dtype=None)


data_2 = np.genfromtxt('type_111_113_133_315.txt', delimiter=',', dtype=None)


data_3 = np.genfromtxt('type_111_202_131_204.txt', delimiter=',', dtype=None)


data_4 = np.genfromtxt('type_111_131_313_204.txt', delimiter=',', dtype=None)


#data_1=data_1.tolist()
data_5 = np.genfromtxt('type_111_202_131_311.txt', delimiter=',', dtype=None)

#data_2=data_2.tolist()
data_6 = np.genfromtxt('type_200_131_113_204.txt', delimiter=',', dtype=None)

#data_4=data_4.tolist()
#with open('type_111_202_131_204.txt') as myfile:
#    data = [next(myfile) for x in xrange(N)]
#scaler = MinMaxScaler()
#MinMaxScaler(copy=True, feature_range=(0, 1))
#for dat in data:
#    for temp_data in dat:
#        temp_data=temp_data/360
num_classes = 6
num_of_samples = len(data_1)+len(data_2)+len(data_3)+len(data_4)+len(data_5)+len(data_6)
data= np.concatenate((data_1, data_2,data_3,data_4, data_5,data_6), axis=0)
label = np.ones((num_of_samples,), dtype='int64')#n=0
n=0
m=0
while(m<= len(data_1)):
    label[n]=0
    n+=1
    m+=1
n=n-1
m=0
while(m<= len(data_2)):
    label[n]=1
    n+=1
    m+=1
n=n-1
m=0    
while(m<= len(data_3)):
    label[n]=2
    n+=1
    m+=1
n=n-1
m=0    
while(m<= len(data_4)):
    label[n]=3
    n+=1
    m+=1
    
n=n-1
m=0 
while(m<= len(data_5)):
    label[n]=4
    n+=1
    m+=1   
n=n-1
m=0 
while(m< len(data_6)):
    label[n]=5
    n+=1
    m+=1      
#    i+=1    
##i=i-1
#for i in range(len(data_2), len(data_3)):
#    label[i]=2
##    i+=1    
##i=i-1
#for i in range(len(data_3), len(data_4)):
#    label[i]=3
##    i+=1    

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam

Y = np_utils.to_categorical(label, num_classes)

#Shuffle the dataset
x,y = shuffle(data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


learning_rate = 1.e-4  #e1e-3 does not work
model = Sequential()
model.add(Dense(128,input_dim=4))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(optimizer=Adam(learning_rate),
              loss='categorical_crossentropy', #MSE,
              metrics=['accuracy'])
#model.fit(xdata , ydata , epochs =200)
model.summary()
hist = model.fit(X_train, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(X_test, y_test))
model.save('Angle_based_try_6ok3.h5') 
scores = model.evaluate( X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
from matplotlib import pyplot as plt

plt.style.use('seaborn-white')
#plt.style.use('ggplot')  its nice!
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('new2{}.png'.format('model accuracy'), bbox_inches='tight', dpi=1000, transparent=False, facecolor='w', edgecolor='w')

# summarize history for loss

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('new2{}.png'.format('model loss'), bbox_inches='tight', dpi=1000, transparent=False,facecolor='w', edgecolor='w')
plt.show()

import csv

with open("history_acc.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in hist.history['acc']:
        writer.writerow([val]) 
        
with open("history_acc_val.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in hist.history['val_acc']:
        writer.writerow([val])       
        
with open("history_loss_val.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in hist.history['val_loss']:
        writer.writerow([val])   

with open("history_loss.txt", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in hist.history['loss']:
        writer.writerow([val])             
#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#from keras.utils import np_utils
#from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
#from keras.layers import Dense, Activation
#from keras.layers import Dropout, 
#from keras.models import Sequential

#N = 40 # number of trials
#search = {
#        'batch_size ': np.random.choice ([16 , 32, 64, 128 , 256] , N),
#        'num_neurons': np.random.choice ([8 , 32, 128 , 512 , 1024] , N),
#        'learn_rate ': np.random.choice ([ -5, -4, -3, -2, -1], N),
#        'activation ': np.random.choice ([ 'relu ', 'elu ', 'sigmoid ', 'tanh '], N),
#        'dropout ': np. random . choice ([0.0 , 0.1 , 0.2 , 0.3 , 0.5 , 0.6] , N),
#        'val_acc ': np. zeros (N)}
#for i in range (N):
#    model = Sequential ([
#    Dense (search ['num_neurons '][i], input_shape =(4 ,) ),
#    Activation ( search ['activation '][i]) ,
#    Dropout ( search ['dropout '][i]) ,
#    Dense (10 , activation ='softmax ')])
#    model.compile(
#    loss ='categorical_crossentropy ',
#    optimizer = Adam (lr =10.** search ['learn_rate '][i]) ,
#    metrics =[ 'accuracy '])
#
#
#for i in range (10) :
## randomly choose 4 sets of trials
#    idx = np.random.choice(N, 4*(i+1) , replace = False )
#    idx = np.array(np.split(idx , 4))
#    acc = np.max(search ['val_acc'][idx], axis =1) # best acc. in each set
#    acc_mean [i] = np.mean(acc) # mean of best accuracies
#    acc_std [i] = np.std (acc , ddof=1) # std of best accuracies

#print(da)
# prepare data for normalization
#values = series.values
#values = values.reshape((len(values), 1))
# train the normalization
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(values)
#print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
## normalize the dataset and print the first 5 rows
#normalized = scaler.transform(values)
#for i in range(5):
#	print(normalized[i])
## inverse transform and print the first 5 rows
#inversed = scaler.inverse_transform(normalized)
#for i in range(5):
#	print(inversed[i])
    
    
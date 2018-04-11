# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:30:35 2018

@author: y.wei
"""

import itertools

import numpy as np
#from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
np.random.seed(7)
#data = np.loadtxt('type_111_202_131_204.txt')
data_1 = np.genfromtxt('type_200_202_131_113.txt', delimiter=',', dtype=None)
#data_1 =data_1[500:1000]

data_2 = np.genfromtxt('type_111_113_133_315.txt', delimiter=',', dtype=None)
#data_2 =data_2[500:1000]

data_3 = np.genfromtxt('type_111_202_131_204.txt', delimiter=',', dtype=None)
#data_3 =data_3[500:1000]

data_4 = np.genfromtxt('type_111_131_313_204.txt', delimiter=',', dtype=None)
#data_4 =data_4[500:1000]

#data_1=data_1.tolist()
data_5 = np.genfromtxt('type_111_202_131_311.txt', delimiter=',', dtype=None)
#data_5 =data_5[500:1000]
#data_2=data_2.tolist()
data_6 = np.genfromtxt('type_200_131_113_204.txt', delimiter=',', dtype=None)
#data_6 =data_6[500:1000]
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

from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
#plt.style.use('ggplot')  its nice!
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 16
Y = np_utils.to_categorical(label, num_classes)

#Shuffle the dataset
x,y = shuffle(data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from keras.models import load_model
model = load_model('Angle_based_try_6ok2.h5')

y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
class_names=['Class A', 'Class B', 'Class C',
              'Class D','Class E','Class F']#,'type_111_113_133_315'
#cnf_matrix = confusion_matrix(y_test, y_pred)
#np.set_printoptions(precision=
y_pred_name=[]
y_pred.tolist()
for y in y_pred:
#    print(y[0])
    maxidx=np.argmax(y[:])
    y_pred_name.append(class_names[maxidx])
    
y_test_name=[]
y_test.tolist()
for y in y_test:
#    print(y[0])
    maxidx=np.argmax(y[:])
    y_pred_name.append(class_names[maxidx])    
    
cnf_matrix = confusion_matrix(y_pred_name, y_pred_name)    
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.savefig('confusion{}.png'.format('C'), bbox_inches='tight', dpi=1000, transparent=False,facecolor='w', edgecolor='w')

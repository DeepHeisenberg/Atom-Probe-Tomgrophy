# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:35:15 2018

@author: y.wei
"""
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
np.random.seed(7)
#data = np.loadtxt('type_111_202_131_204.txt')
history_acc = np.genfromtxt('history_acc.txt', delimiter=',', dtype=None)
#data_1=data_1.tolist()
history_acc_val= np.genfromtxt('history_acc_val.txt', delimiter=',', dtype=None)
#data_2=data_2.tolist()
history_loss = np.genfromtxt('history_loss.txt', delimiter=',', dtype=None)
#data_3=data_3.tolist()
history_loss_val = np.genfromtxt('history_loss_val.txt', delimiter=',', dtype=None)

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
#plt.style.use('ggplot')  its nice!
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 18

plt.figure()
plt.plot(history_acc)
plt.plot(history_acc_val)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('new2{}.png'.format('model accuracy'), bbox_inches='tight', dpi=1000, transparent=False, facecolor='w', edgecolor='w')

# summarize history for loss
plt.figure()
plt.plot(history_loss)
plt.plot(history_loss_val)
plt.title('Cross-entropy Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('new2{}.png'.format('model loss'), bbox_inches='tight', dpi=1000, transparent=False,facecolor='w', edgecolor='w')
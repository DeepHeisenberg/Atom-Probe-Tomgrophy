# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:04:16 2018

@author: y.wei
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
#count, bins, ignored = plt.hist(s1, 30, normed=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
#plt.show()

#mu3, sigma3 = 71, 4
#s3 = np.random.normal(mu3, sigma3, 1000)
#count, bins, ignored = plt.hist(s3, 30, normed=True)
#plt.plot(bins, 1/(sigma3 * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu3)**2 / (2 * sigma3**2) ), linewidth=2, color='r')
#plt.show()
#mu3, sigma3 =  116, 1
#s3 = np.random.normal(mu3, sigma3, 1000)
#count, bins, ignored = plt.hist(s3, 30, normed=True)
##plt.figure()
#plt.plot(bins, 1/(sigma3 * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu3)**2 / (2 * sigma3**2) ), linewidth=2, color='r')
#plt.show()

N=10000
n=0
sigma =2
type_200_131_113_204 = []
while(n<=N):
#    mu1, sigma1 = 88, 4
#    s1 = np.random.normal(mu1, sigma1, 1)
#    mu2, sigma2 = 71, 4
#    s2 = np.random.normal(mu2, sigma2, 1)
#    mu3, sigma3 = 130, 2
#    s3 = np.random.normal(mu3, sigma3, 1)

    mu2, sigma2 = 71, sigma
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 = 71, sigma
    s3 = np.random.normal(mu3, sigma3, 1)
    mu1, sigma1 = 88, sigma
    s1 = np.random.normal(mu1, sigma1, 1)
    mu4, sigma4 = 130, sigma
    s4 = np.random.normal(mu4, sigma4, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]
    if( total<=360 and total>=358):
        items=[s3[0], s2[0],s1[0],s4[0]]
        type_200_131_113_204.append(items)
        n+=1
with open('type_200_131_113_204', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(type_200_131_113_204)    


type_200_202_131_113 = []
n=0
while(n<=N):
    mu2, sigma2 = 71, sigma
    s2 = np.random.normal(mu2, sigma2, 1)
    mu1, sigma1 = 88, sigma
    s1 = np.random.normal(mu1, sigma1, 1)
    mu3, sigma3 = 104, sigma
    s3 = np.random.normal(mu3, sigma3, 1)
    mu4, sigma4 = 102, sigma
    s4 = np.random.normal(mu4, sigma4, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]
    if( total<=360 and total>=358):
        items=[s2[0],s1[0],s3[0],s4[0]]
        type_200_202_131_113.append(items)
        n+=1
with open('type_200_202_131_113', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(type_200_202_131_113)    
    
    
    
type_111_202_131_204 = []
n=0
while(n<=N):
    mu1, sigma1 = 57, sigma
    s1 = np.random.normal(mu1, sigma1, 1)
    mu2, sigma2 = 86, sigma
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 = 104, sigma
    s3 = np.random.normal(mu3, sigma3, 1)
    mu4, sigma4 = 110, sigma
    s4 = np.random.normal(mu4, sigma4, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]
    if( total<=360 and total>=358):
        items=[s1[0],s2[0],s3[0],s4[0]]
        type_111_202_131_204.append(items)
        n+=1
with open('type_111_202_131_204', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(type_111_202_131_204)   

type_111_202_131_311 = []
n=0
while(n<=N):
    mu1, sigma1 = 70, sigma
    s1 = np.random.normal(mu1, sigma1, 1)
    mu2, sigma2 = 70, sigma
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 = 106, sigma
    s3 = np.random.normal(mu3, sigma3, 1)
    mu4, sigma4 = 116, sigma
    s4 = np.random.normal(mu4, sigma4, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]
    if( total<=360 and total>=358):
        items=[s1[0],s2[0],s3[0],s4[0]]
        type_111_202_131_311.append(items)
        n+=1
with open('type_111_202_131_311', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(type_111_202_131_311)   
#
type_111_113_133_315 = []
n=0
while(n<=N):
    mu1, sigma1 = 57, sigma
    s1 = np.random.normal(mu1, sigma1, 1)
    mu2, sigma2 = 77, sigma
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 = 98, sigma
    s3 = np.random.normal(mu3, sigma3, 1)
    mu4, sigma4 = 126, sigma
    s4 = np.random.normal(mu4, sigma4, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]
    if( total<=360 and total>=358):
        items=[s1[0],s2[0],s3[0],s4[0]]
        type_111_113_133_315.append(items)
        n+=1
with open('type_111_113_133_315', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(type_111_113_133_315)   
 
type_111_131_313_204 = []
while(n<=N):
#    mu1, sigma1 = 88, 4
#    s1 = np.random.normal(mu1, sigma1, 1)
#    mu2, sigma2 = 71, 4
#    s2 = np.random.normal(mu2, sigma2, 1)
#    mu3, sigma3 = 130, 2
#    s3 = np.random.normal(mu3, sigma3, 1)

    mu2, sigma2 = 58, sigma
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 = 78, sigma
    s3 = np.random.normal(mu3, sigma3, 1)
    mu1, sigma1 = 108, sigma
    s1 = np.random.normal(mu1, sigma1, 1)
    mu4, sigma4 = 114, sigma
    s4 = np.random.normal(mu4, sigma4, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]
    if( total<=360 and total>=358):
        items=[s2[0], s3[0],s1[0],s4[0]]
        type_111_131_313_204.append(items)
        n+=1
with open('type_111_131_313_204', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(type_111_131_313_204)    
    
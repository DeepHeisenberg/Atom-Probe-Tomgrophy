#import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#sns.set(style="ticks")
#from sklearn import datasets
#from sklearn.decomposition import PCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
#np.random.seed(7)
##data = np.loadtxt('type_111_202_131_204.txt')
data_1 = np.genfromtxt('type_111_202_131_204.txt', delimiter=',', dtype=None)
data_1 =data_1[:500]
#data_1=data_1.tolist()
data_2 = np.genfromtxt('type_111_202_131_311.txt', delimiter=',', dtype=None)
data_2 =data_2[:500]
#data_2=data_2.tolist()
data_3 = np.genfromtxt('type_200_131_113_204.txt', delimiter=',', dtype=None)
data_3 =data_3[:500]
#data_3=data_3.tolist()
data_4 = np.genfromtxt('type_200_202_131_113.txt', delimiter=',', dtype=None)
data_4 =data_4[:500]

data_5 = np.genfromtxt('type_111_113_133_315.txt', delimiter=',', dtype=None)
data_5 =data_5[:500]
target_names=['type_111_202_131_204', 'type_111_202_131_311', 'type_200_131_113_204', 'type_200_202_131_113','type_111_113_133_315']
#data_4=data_4.tolist()
#with open('type_111_202_131_204.txt') as myfile:
#    data = [next(myfile) for x in xrange(N)]
#scaler = MinMaxScaler()
#MinMaxScaler(copy=True, feature_range=(0, 1))
#for dat in data:
#    for temp_data in dat:
#        temp_data=temp_data/360
num_classes = 5
num_of_samples = len(data_1)+len(data_2)+len(data_3)+len(data_4)+len(data_5)
data= np.concatenate((data_1, data_2,data_3,data_4, data_5), axis=0)
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
while(m<=len(data_4)):
    label[n]=3
    n+=1
    m+=1
    
n=n-1
m=0 
while(m< len(data_4)):
    label[n]=4
    n+=1
    m+=1   



import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
#from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
X = data
y = label
#target_names = names

#pca = PCA(n_components=2)
#
#X_r = pca.fit(X).transform(X)
#
#lda = LinearDiscriminantAnalysis(n_components=2)
#X_r2 = lda.fit(X, y).transform(X)

X_r = TSNE(n_components=2).fit_transform(X)

## Percentage of variance explained for each components
#print('explained variance ratio (first two components): %s'
#      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange','palegreen','firebrick']
lw = 2
#from mpl_toolkits.mplot3d import Axes3D

for color, i, target_name in zip(colors, [0, 1, 2,3,4], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
#    g = sns.jointplot(x="x", y="y", data=X_r[y==1], kind="kde", color="m")
#    g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
#    g.ax_joint.collections[0].set_alpha(0)
#    g.set_axis_labels("$X$", "$Y$");
plt.legend(loc='best', shadow=False, scatterpoints=1)
title='PCA of quadrangle database'
plt.title(title)
plt.xlabel('Principle component 1')
plt.ylabel('Principle component 2')

#plt.figure()
#for color, i, target_name in zip(colors, [0, 1, 2,3,4], target_names):
#    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
#                label=target_name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
#plt.title('LDA of quadrangle database')

plt.show()
#from matplotlib import pyplot as plt
#
##plt.title(title)
# 
plt.savefig('detection11{}.pdf'.format(title), bbox_inches='tight')
#X_pca = PCA(n_components=2).fit_transform(d2_train_dataset)
## plot the result
#plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=label)
#plt.colorbar(ticks=range(10))
#
#plt.show()
#print(imgs.shape)
#imgs /=255 
#nsamples, nx, ny = imgs.shape
#d2_train_dataset = imgs.reshape((nsamples,nx*ny))
##X_tsne = TSNE(learning_rate=100).fit_transform(d2_train_dataset)
#X_pca = PCA(n_components=2).fit_transform(d2_train_dataset)
#plt.figure(figsize=(10, 5))
##plt.subplot(121)
##plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='red')
##plt.subplot(122)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue')
#
#fig = plt.figure(1, figsize=(4, 3))
#plt.clf()
#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
#plt.cla()
#pca = decomposition.PCA(n_components=3)
#pca.fit(X)
#X = pca.transform(X)
#
#for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
#    ax.text3D(X[y == label, 0].mean(),
#              X[y == label, 1].mean() + 1.5,
#              X[y == label, 2].mean(), name,
#              horizontalalignment='center',
#              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
## Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(np.float)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
#           edgecolor='k')
#
#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])
#
#plt.show()

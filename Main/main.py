# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:52:27 2018

@author: y.wei
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:42:36 2018

@author: y.wei
"""

import numpy as np
import cv2
#from itertools import  permutations
from itertools import combinations
#import time
#from directkeys import ReleaseKey, PressKey, W, A, S, D
#import pyautogui
from matplotlib import pyplot as plt
from pole_detection import Pole_detection   

def magnitude(p1, p2):
    vect_x = p2[0] - p1[0]
    vect_y = p2[1] - p1[1]
    return np.sqrt(vect_x**2 + vect_y**2)


def find_closest_points(mypole, poles, num):
    dis=[]
    for x in range(0,len(poles)):
        dis.append(magnitude(mypole, poles[x]))
    minimum = sorted(dis)
    indices=[]
    i=0
    for i in range(0,num):
      indice = [j for j, v in enumerate(dis) if v == minimum[i+1]]
      indices.append(indice[0])
    closest_total_dis= sum(dis[:num])
    return indices, closest_total_dis

def find_angles(mypole, poles):   
    a = np.array(poles[0])
    b = np.array(mypole)
    c = np.array(poles[1])
    ba = a - b
    bc = c - b
    cosine_angle_ac = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_ac = np.arccos(cosine_angle_ac)
    return np.degrees(angle_ac)

def draw_poles(img, pole, pole_size,colors):
    pole=[np.int(pole[0]), np.int(pole[1])]
    new_image = cv2.circle(img, (pole[0],
    pole[1]),
    radius=np.int(pole_size), 
    color=colors,
    thickness=5)
    return new_image


def find_symmetry(mypole, poles):
    dis=[]
    relatve_difference =[]
    relate_poles=[]
    for pole in poles:
        dis.append(magnitude(mypole, pole))
        
    d=list(combinations(dis, 2))
    for dd in d:
        d0=abs(dd[0]-dd[1])/dd[1]
        relatve_difference.append(d0)
    return relatve_difference, relate_poles

def find_poles_symmetry(mypole, poles):
    
    comb_poles = list(combinations(poles, 2))
  
    sym_poles =[]
    for poles in comb_poles:
         d1 = magnitude(mypole, poles[0])
         d2 = magnitude(mypole, poles[1])
         d0 = abs(d1-d2)/d1
         if d0 <= 0.12:
             sym_poles.append(poles)
    return sym_poles


def draw_sym_poles(img, mypole, pole):
    pole0=[np.int(mypole[0]), np.int(mypole[1])]
    pole1=[np.int(pole[0][0]), np.int(pole[0][1])]
    pole2=[np.int(pole[1][0]), np.int(pole[1][1])]
    new_image =  cv2.line(img, (pole0[0],pole0[1]), (pole1[0],pole1[1]), [255,255,255], 3)
    new_image =  cv2.line(new_image, (pole0[0],pole0[1]), (pole2[0],pole2[1]), [255,255,255], 3)    
    return new_image

def angle_sampling(i, poles):
    angles=[]  
    mypole = poles[i]
    del poles[i]
    for j in list(combinations(poles, 2)):
        angle =find_angles(mypole, j) 
        angles.append(angle)
    return angles
  
def PolygonArea(poles):
    Cs=[]
    for pole in list(combinations(poles, 2)):   
         C=magnitude(pole[0], pole[1])
         Cs.append(C)
    total_C=sum(Cs)         
    return total_C

def find_angles_polygon(poles):
    angles = []
    for pole in poles:    
        maximum_angle=1
        for i in list(combinations(poles, 2)):
            angle =find_angles(pole, i) 
            if maximum_angle <= angle:
                maximum_angle = angle
        angles.append(maximum_angle)
    return angles  

      
def check_polygon_length_ratio(poles):
    ds=[]
    x=list(combinations(poles, 2))
    for xx in x:
         d1 = magnitude(xx[0], xx[1])
         ds.append(d1)
    ratio = min(ds)/max(ds)
    return ratio, max(ds)
         
    
if __name__ == '__main__': 


    FileName = 'N26.png'
    screen = cv2.imread(FileName) 
    height,width,channels = screen.shape   
    minThres=1
    maxThres=200
    minArea=10
    erosion=4
    dilation =7
    a=Pole_detection(FileName, erosion, dilation, minArea, minThres, maxThres)
    poles, poles_size =a.Blob_detector()




    comp_image = draw_poles(screen,  poles[1], poles_size[1], (0,0,255))
    i=0
    ii=0
    poles_high_rank =[]
    poles_size_high_rank=[]
    num_shown_poles=6
    for pole, pole_size in zip(poles, poles_size):   
        if screen[np.int(pole[1]), np.int(pole[0]),0]>20:          
            poles_high_rank.append(poles[ii])
            poles_size_high_rank.append(pole_size)
            i+=1 
        ii+=1
        if i>=num_shown_poles:
            break        
        
    j=0        
    for pole, pole_size in zip(poles_high_rank, poles_size_high_rank):    
            comp_image = draw_poles(comp_image, pole, pole_size,(255,255,255))
        #            for x in range(0,len(keypoints)):
            text='(%s)' % (j)
            location=(np.int(pole[0])-2*np.int(pole_size), 
                      np.int(pole[1])- np.int(pole_size))
            j+=1
            
            
    num_adjacent_poles=3
    angles=[]
    my_true_poles=[]
    rank_true_poles=[]
    
    m=0 
    #high_rank = 6
    for  mypoles, rank in zip(list(combinations(poles_high_rank, 4)), list(combinations(list(range(0,num_shown_poles)), 4))):    
    #     high_rank=100 # any big value is fine   
         angles = find_angles_polygon(mypoles)
    
         indices = [i for i, x in enumerate(angles) if x >=115]
         
         if len(indices)<=1:
             if(max(angles)<=160 and min(angles)>=15 and sum(angles)>358):   
                current_rank=sum(rank)
                my_true_poles.append(mypoles) 
                rank_true_poles.append(current_rank)
    #            text_X=np.array([sorted(find_angles_polygon(mypoles))])
    #            preds = model.predict(text_X)
    #            maxidx=np.argmax(preds[0][:])
    #            print(' this pole {} is a {} with confidence: {:.2f}%'
    #                  .format(text_X, names[maxidx],preds[0][maxidx] * 100))
         else:
             print('your quadrangle is too squeezed!')
             
    
                
    final_true_poles = []
    ratio_final_true_poles = []
    ratio_Q_factor=0.455
    size_maxratio=0.75
    for pole in my_true_poles:
        r,maxd=check_polygon_length_ratio(pole)
        size=maxd/width
        print('ratio:{}'.format(r))
        print('size:{}'.format(size))
        if r > ratio_Q_factor and  size < size_maxratio:
            final_true_poles.append(pole)  
            ratio_final_true_poles.append(r)
            print('{} bigger than ratio Q factor {} and its size {} is small than 0.8, qualified!'.format(r,ratio_Q_factor, size))
    if len(final_true_poles)==0:
        print('we have failed to find a qualified polygon!')
        
             
    unique_true_poles=[]
    new_Areas=[]    
    final_angles=[]
    if len(final_true_poles)>0:
        for mypole in final_true_poles:
    #            corners_sorted = sorted(mypole)
                new_Area = PolygonArea(mypole)
                final_angle = find_angles_polygon(mypole)
    #            new_Area=PolygonArea(mypole)
                new_Areas.append(new_Area)
                final_angles.append(final_angle)
        tempo_true_angles=[]
        tempo_true_types=[]
        tempo_true_poles=[]
        tempo_preds=[]
        final_areas=[]
        from test_OOP import Classification
        from keras.models import load_model
        model = load_model('Angle_based_try_6ok3.h5')
        for my_angles, mypoles in zip(final_angles, final_true_poles):
             print('angle sampling and selection:')
             new_a=Classification(my_angles,model)
             tempo_type,preds=new_a.neural_network()
             print(preds)
             
             
    #         tempo_type= check_angles(my_angles)
    #          tempo_poles= mypoles
    
             indice_none_zero = [i for i, x in enumerate(tempo_type) if (x !=0)]
             if len(indice_none_zero)>=2:
    #            min_value = min(new_Areas)
    #            min_index = new_Areas.index(min_value)
    #            unique_true_poles=final_true_poles[min_index]    
                 tempo_true_angles.append(my_angles)
                 tempo_true_types.append(tempo_type)
                 tempo_true_poles.append(mypoles)
                 tempo_preds.append(preds)
                 print(my_angles)  
                 print(sum(my_angles))
    #             text_X=np.array([sorted(my_angles)])
    #             preds = model.predict(text_X)
    #             maxidx=np.argmax(preds[0][:])
    #             print(' this pole {} is a {} with confidence: {:.2f}%'
    #                   .format(text_X, names[maxidx],preds[0][maxidx] * 100))
    #         if len(indice_none_zero)==4:
    ##             Areaw =PolygonArea(mypole)
    ##             if minimum_Area
    #             = my_angles
    #            break
                 
                
    #             minimum_Area=0
    
        if len(tempo_true_types)>=1:
            for mypole in tempo_true_poles:
        #            corners_sorted = sorted(mypole)
                final_area = PolygonArea(mypole)
                final_areas.append(final_area)
    
            minimum = sorted(final_areas)
            mimimum_indice = [j for j, v in enumerate(final_areas) if v == minimum[0]]
    #        mimimum_indice[0]=3
            true_types=tempo_true_types[mimimum_indice[0]]
            true_angles=tempo_true_angles[mimimum_indice[0]]
            unique_true_poles=tempo_true_poles[mimimum_indice[0]]
            
    
                    
    #    if true_types[0]>0: 
            for pole, angle,true_type in zip(unique_true_poles, true_angles, true_types):    
                    comp_image = draw_poles(comp_image, pole, 10, (0,0,0))
                    text_type='(type:{}'.format(true_type)
                    text_angle='(angle:{})'.format( np.int(angle))
                    text_position = '(X:{}, Y:{})'.format(np.int(pole[0])-width/2,-(np.int(pole[1])-height/2))
                    location_type=(np.int(pole[0])-1*np.int(pole_size), 
                              np.int(pole[1])+2*np.int(pole_size))
                    location_position=(np.int(pole[0])-3*np.int(pole_size), 
                                       np.int(pole[1])+4*np.int(pole_size))
                    comp_image=cv2.putText(
                                        comp_image, text_type+text_angle, location_type,
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.8,
                                        (255,0,0),
                                       2) 
                    comp_image=cv2.putText(
                        comp_image, text_position, location_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8,
                        (0,255,0),
                       2)
        else:
            min_value = min(new_Areas)
            min_index = new_Areas.index(min_value)
            unique_poles=final_true_poles[min_index]    
            unique_angles = find_angles_polygon(unique_poles) 
    #        text_X=np.array([sorted(find_angles_polygon(unique_poles))])
    #        preds = model.predict(text_X)
    #        maxidx=np.argmax(preds[0][:])
    #        print(' this pole {} is a {} with confidence: {:.2f}%'
    #              .format(text_X, names[maxidx],preds[0][maxidx] * 100))
            the_type='unknown'        
            for pole, angle in zip(unique_poles, unique_angles):    
                    comp_image = draw_poles(comp_image, pole, 10, (255,0,255))
                    text_angle='(angle:{})'.format(np.int(angle))
    
                    location=(np.int(pole[0])-2*np.int(pole_size), 
                              np.int(pole[1])+2*np.int(pole_size))
    #                comp_image=cv2.putText(
    #                                    comp_image, text_angle,location,
    #                                    cv2.FONT_HERSHEY_SIMPLEX, 
    #                                    0.8,
    #                                    (0,0,0),
    #                                   3)
            new_a=Classification(unique_angles,model)
            tempo_type,preds=new_a.neural_network() 
    title = FileName
    plt.figure(figsize=(8, 6))
    plt.imshow(comp_image)
    #plt.title(title)
    plt.axis('off') 
    plt.savefig('clear{}.png'.format(title), bbox_inches='tight', dpi=1000, transparent=False)
    
    def prediction_bar_chart():
        
        plt.style.use('seaborn-white')
    #plt.style.use('ggplot')  its nice!
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 18
        plt.rcParams['figure.titlesize'] = 16
    #    plt.rcParams['figure.figsize'] = 50, 40
    #    preds=tempo_preds[inx].tolist()
        #preds = [i for i in preds]
        y=preds[0]
        y= [ round(elem, 2) for elem in y ]
        #variance = [1, 2, 7, 4, 2, 3]
        names = ['Class A','Class B','Class C',
                 'Class D','Class E','Class F']
        labels = ['{200}-{202}-{113}-{113}','{111}-{113}-{133}-{135}','{111}-{202}-{113}-{204}',
                 '{111}-{131}-{133}-{204}','{111}-{202}-{113}-{113}','{200}-{113}-{113}-{204}']
        #x = [u'INFO', u'CUISINE', u'TYPE_OF_PLACE', u'DRINK', u'PLACE', u'MEAL_TIME']
        fig, ax = plt.subplots(figsize=(8, 6))    
        width = 0.75 # the width of the bars 
        ind = np.arange(len(y))  # the x locations for the groups
        mind=np.argmax(y)
        colors =[]
        for i,j in enumerate(y):
            if i ==mind:
                color = 'palegreen'
                colors.append(color)
            else:
                color ='palegreen'
                colors.append(color)
                
        ax.barh(ind, y, width, color=colors)
         
        for i, v in enumerate(y):
            ax.text(v, i+.25, str(v), color='red', fontweight='bold')
            ax.text(-.17, i-.15, names[i], color='black', fontweight='bold')
            ax.text(0.05, i-.15, labels[i], color='black', fontweight='bold')
            
        ax.set_yticks(ind+width/2)
        ax.set_yticklabels([])
        ax.set_xlim(0, 1) 
        plt.xlabel('Probability')
    #    plt.ylabel('y')      
        #plt.show()
        plt.savefig('test{}.png'.format('olala'), dpi=1000, format='png', bbox_inches='tight')
        plt.show()
    prediction_bar_chart()    
#print(find_angles(poles_high_rank[5],[poles_high_rank[1],poles_high_rank[6]])) 
#new_poles=poles[0:6]
#del new_poles[3]
#x=angle_sampling(3, poles[:6])
#print(sorted(x))
#cv2.putText(comp_image, 
#            text,location,
#            cv2.FONT_HERSHEY_SIMPLEX, 
#            0.5,
#            (0,0,0),
#           2)    
#    print(total_dis)

#pole_angles=[]
#for i in list(combinations(poles[:4], 2)):
#    new_poles, total_dis = find_closest_points(poles, poles[:4], num_adjacent_poles)
#    angle= find_angles(mypole, i)
    
#    if angle >= 90:
#        angle =180-angle
#    pole_angles.append(angle)
#for index in range(0,1):


#[[  65.50197196   65.93812145   88.69047895  139.86942764]]
# [[  65.32658545   65.46019189   88.31928899  140.89393368]]
# [[  64.96818606   65.18299589   88.54459703  141.30422102]]
    
    
    
#x= np.array([unique_true_poles[0][0] ,unique_true_poles[1][0] ,unique_true_poles[2][0] ,unique_true_poles[3][0]]) 
#y= np.array([unique_true_poles[0][1] ,unique_true_poles[1][1] ,unique_true_poles[2][1] ,unique_true_poles[3][1]]) 
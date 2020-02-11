# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:47:59 2018

@author: y.wei
"""
import numpy as np
import cv2
#from Angle_based_OOP import process_img


class Pole_detection:
    
    def __init__(self,FileName,erosion, dilation,minArea ,minThres ,maxThres ): # minArea, minThres, maxThres, 
        self.FileName = FileName
        self.minArea = minArea
        self.minThres = minThres
        self.maxThres = maxThres
        self.erosion = erosion
        self.dilation = dilation
    def denoise(self):
        image = cv2.imread(self.FileName)
        hsv = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))#np.invert
        img_new = cv2.medianBlur(hsv,9)
        blurred = cv2.GaussianBlur(img_new, (11, 11), 0)
        thresh1 = cv2.erode(blurred, None, iterations=self.erosion)
        thresh2 = cv2.dilate(thresh1, None, iterations=self.dilation)
        return thresh2  
    def Blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
    #     
        # Change thresholds
        params.minThreshold = self.minThres
        params.maxThreshold = self.maxThres
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = self.minArea  # 2,200, 200,0.1,0.1,0.1
        ## 
        ### Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        ## 
        ## Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1 #0.3
        # 
        ## Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1 #0.3
         
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)        
        keypoints = detector.detect(self.denoise())
        if keypoints is None:
            print('Poles is not found!')
            return []  
        else:
            coords_keypoints = [list(p.pt) for p in keypoints]
            keysize = [np.int(p.size) for p in keypoints]
    
            return coords_keypoints, keysize
#    def line_detection(self):
        
class Classification:       
    
      def __init__(self, angles, model):
          self.angles = angles
          self.model =model
      def neural_network(self):
        names = [' A: type_200_202_131_113','B: type_111_113_133_315','C: type_111_202_131_204',
                 'D: type_111_131_313_204','E: type_111_202_131_311','F: type_200_131_113_204']
        new_angles = sorted(self.angles)
        text_X=np.array([new_angles])
        preds = self.model.predict(text_X)
        maxidx=np.argmax(preds[0][:])
        print(' this pole {} is a {} with confidence: {:.2f}%'
              .format(text_X, names[maxidx],preds[0][maxidx] * 100))
        true_types=[0,0,0,0]
    #    indice_84_92 = [i for i, x in enumerate(true_angles) if (x >=84 and x<=92)]
    #    indice_67_75 = [i for i, x in enumerate(true_angles) if (x >=67 and x<=75)]
        indice_100_109 = [i for i, x in enumerate(self.angles) if (x >=100 and x<=109)]
    #    indice_99_105 = [i for i, x in enumerate(true_angles) if (x >=99 and x<=105)]  
    #    indice_below_62 = [i for i, x in enumerate(true_angles) if (x <=63)] 
    #    indice_beyond_120 = [i for i, x in enumerate(true_angles) if (x >=120)]
    #    indice_beyond_100= [i for i, x in enumerate(true_angles) if (x >=100)]
    #    indice_93_102 = [i for i, x in enumerate(true_angles) if (x >=93 and x<=102)] 
        if preds[0][maxidx]>0.999:
            if ( maxidx==5):
                if max(new_angles)>136:
                    print(names[maxidx])
                    print('200 is {} \n.'.format(new_angles[2]))
                    print('204 is {} \n.'.format(new_angles[0]))
                    print('402 is {} \n.'.format(new_angles[1]))
                    print('113 is {} \n.'.format(new_angles[3]))
                    for n,i in enumerate(self.angles):
                        if i==new_angles[2]:
                          true_types[n]=200
                        if i==new_angles[0]: 
                          true_types[n]=204
                        if i==new_angles[1]: 
                          true_types[n]=402
                        if i==new_angles[3]: 
                          true_types[n]=113                   
                else:                
                    print(names[maxidx])
                    print('200 is {} \n.'.format(new_angles[2]))
                    print('131 is {} \n.'.format(new_angles[0]))
                    print('113 is {} \n.'.format(new_angles[1]))
                    print('204 is {} \n.'.format(new_angles[3]))
                    for n,i in enumerate(self.angles):
                        if i==new_angles[2]:
                          true_types[n]=200
                        if i==new_angles[0]: 
                          true_types[n]=131
                        if i==new_angles[1]: 
                          true_types[n]=113
                        if i==new_angles[3]: 
                          true_types[n]=204   
            if (maxidx==1):
                print(names[maxidx])
                print('111 is {} \n.'.format(new_angles[0]))
                print('313 is {} \n.'.format(new_angles[1]))
                print('315 is {} \n.'.format(new_angles[3]))
                print('113 is {} \n.'.format(new_angles[2]))
                for n,i in enumerate(self.angles):
                    if i==new_angles[0]:
                      true_types[n]=111
                    if i==new_angles[1]: 
                      true_types[n]=113
                    if i==new_angles[3]: 
                      true_types[n]=315
                    if i==new_angles[2]: 
                      true_types[n]=313
            if (maxidx==3):
                print(names[maxidx])
                print('111 is {} \n.'.format(new_angles[0]))
                print('204 is {} \n.'.format(new_angles[1]))
                print('113 is {} \n.'.format(new_angles[2]))
                print('313 is {} \n.'.format(new_angles[3]))
                for n,i in enumerate(self.angles):
                    if i==new_angles[0]:
                      true_types[n]=111
                    if i==new_angles[1]: 
                      true_types[n]=204
                    if i==new_angles[2]: 
                      true_types[n]=113
                    if i==new_angles[3]: 
                      true_types[n]=313
            elif (maxidx==4):    
                print( names[maxidx])
                print('111 is {} \n.'.format(new_angles[3]))
                print('131 is {} \n.'.format(new_angles[0]))
                print('113 is {} \n.'.format(new_angles[1]))
                print('202 is {} \n.'.format(new_angles[2]))
                for n,i in enumerate(self.angles):
                    if i==new_angles[3]:
                      true_types[n]=111
                    if i==new_angles[0]: 
                      true_types[n]=131
                    if i==new_angles[1]: 
                      true_types[n]=113
                    if i==new_angles[2]: 
                      true_types[n]=202    
            elif(maxidx==0):
                print( names[maxidx])
                print('200 is {} \n.'.format(new_angles[1]))
                print('131 is {} \n.'.format(new_angles[2]))
                print('113 is {} \n.'.format(new_angles[3]))
                print('202 is {} \n.'.format(new_angles[0]))
                for n,i in enumerate(self.angles):
                    if i==new_angles[1]:
                      true_types[n]=200
                    if i==new_angles[2]: 
                      true_types[n]=131
                    if i==new_angles[3]: 
                      true_types[n]=113
                    if i==new_angles[0]: 
                      true_types[n]=202
                  
                  
            elif(maxidx==2):
                print( names[maxidx])
                if len(indice_100_109)==2:
                    print('111 is {} \n.'.format(new_angles[0]))
                    print('202 is {} \n.'.format(new_angles[1]))
                    print('204 is {} \n.'.format('unkonwn'))
                    print('113 is {} \n.'.format('unkonwn'))
                    for n,i in enumerate(self.angles):
                        if i==new_angles[0]: 
                          true_types[n]=111
                        if i==new_angles[1]: 
                          true_types[n]=202
                        if i==new_angles[2]: 
                          true_types[n]='unkown'
                        if i==new_angles[3]:
                          true_types[n]='unkonwn'
                else:
                    print('111 is {} \n.'.format(new_angles[0]))
                    print('202 is {} \n.'.format(new_angles[1]))
                    print('204 is {} \n.'.format(new_angles[3]))
                    print('113 is {} \n.'.format(new_angles[2]))
                    for n,i in enumerate(self.angles):
                        if i==new_angles[0]: 
                          true_types[n]=111
                        if i==new_angles[1]: 
                          true_types[n]=202
                        if i==new_angles[3]: 
                          true_types[n]=204
                        if i==new_angles[2]:
                          true_types[n]=113
            
        
        return true_types,preds
          

        
FileName = '8.png'
screen = cv2.imread(FileName) 
height,width,channels = screen.shape   
##plt.figure()
##plt.imshow(screen)
##screen = cv2.flip(screen,0)
##screen = cv2.flip(screen,1)
#minThres=1
#maxThres=200
#minArea=1
#erosion=4
#dilation =7
#lines, poles, poles_size = process_img(screen,minArea, minThres, maxThres, erosion, dilation)
minThres=1
maxThres=200
minArea=1
erosion=4
dilation =7
a=Pole_detection(FileName, erosion, dilation, minArea, minThres, maxThres)

a0=a.denoise()
poles, poles_size =a.Blob_detector()


#new_a=Classification(angles)
#X=new_a.neural_network()





#!/usr/bin/env python
# coding: utf-8

# ### Imports απαιτούμενων πακέτων και βιβλιοθηκών

# In[4]:


import cv2
import numpy as np
import scipy.io
import math
from math import pi
import operator
from numpy.linalg import inv
from numpy.linalg import det
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import map_coordinates
import matplotlib.patches as patches
from matplotlib.pyplot import quiver
from scipy import signal
from scipy import ndimage
from scipy.ndimage import convolve1d
from itertools import chain
from scipy.stats import multivariate_normal


# ### Μέρος 2
# 
# ### Εντοπισμός Χωρο-χρονικών Σημείων Ενδιαφέροντος και Εξαγωγή Χαρακτηριστικών σε Βίντεο Ανθρωπίνων Δράσεων

# ### 2.1 Χωρο-χρονικά Σημεία Ενδιαφέροντος

# In[5]:


import cv20_lab2_2_utils as p
frames = p.read_video('cv20_lab2_part2_material/running/person01_running_d1_uncomp.avi',200,0);
frames = frames.astype(np.float)/255


# ### Βοηθητική συνάρτηση για την εύρεση των πρώτων 500 μεγαλύτερων τιμών του κριτηρίου H και την δημιουργία του πίνακα N × 4 του μέρους 2.1.3

# In[6]:


def Interest_Points(H,sigma):  
    
    sorted_indx = H.flatten().argsort()[::-1][:500]
    dim_idx = np.unravel_index(sorted_indx, H.shape)
    
    x_coord = dim_idx[1].reshape(1,500)
    y_coord = dim_idx[0].reshape(1,500)
    t_coord = dim_idx[2].reshape(1,500)
    
    xyt_coord = np.concatenate((x_coord,y_coord,t_coord),axis=0)
    xyt_coord = xyt_coord.T
    
    scale = np.ones((xyt_coord.shape[0],1))*sigma                   # 1 column array with current scale
    voxels_scale = np.concatenate((xyt_coord, scale), axis=1)       # Concatenate coordinates with scale

    return voxels_scale


# ### 2.1.1  Harris Detector

# In[9]:


def HarrisDetector(frames,sigma,taf,s,k):   
    
    # Spatial Gaussian with scale sigma
    n = int(2*np.ceil(3*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, sigma)
    gauss1D_spatial = list(chain.from_iterable(gauss1D))
    
    # Time Gaussian with scale taf
    n = int(2*np.ceil(3*taf)+1)
    gauss1D = cv2.getGaussianKernel(n, taf)                            
    gauss1D_time = list(chain.from_iterable(gauss1D))
    
    # Image smoothing with Gaussian
    L = convolve1d(frames,gauss1D_spatial,axis = 0, mode= 'reflect')   # Convolve with gaussian kernel in x-dimension
    L = convolve1d(L,gauss1D_spatial,axis = 1, mode= 'reflect')        # Convolve with gaussian kernel in y-dimension
    L = convolve1d(L,gauss1D_time,axis = 2, mode= 'reflect')           # Convolve with gaussian kernel in t-dimension
    
    kernel = [-1,0,1]
    L_x = convolve1d(L,kernel,axis = 0, mode='reflect')  # Calculate Lx by convolving L image with kernel in x-dimension
    L_y = convolve1d(L,kernel,axis = 1, mode='reflect')  # Calculate Ly by convolving L image with kernel in y-dimension
    L_t = convolve1d(L,kernel,axis = 2, mode='reflect')  # Calculate Lt by convolving L image with kernel in t-dimension
    
    # spatial Gaussian with scale s*sigma
    n = int(2*np.ceil(3*s*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, s*sigma)
    gauss1D_spatial = list(chain.from_iterable(gauss1D))
   
    # time Gaussian with scale s*taf
    n = int(2*np.ceil(3*s*taf)+1)
    gauss1D = cv2.getGaussianKernel(n, s*taf)                             
    gauss1D_time = list(chain.from_iterable(gauss1D))
    
    # Calculation of M matrix 
    M11 = convolve1d(convolve1d(convolve1d(L_x**2,gauss1D_spatial,axis=0, mode= 'reflect'),gauss1D_spatial,axis = 1, mode= 'reflect'), gauss1D_time, axis = 2, mode= 'reflect')
    M22 = convolve1d(convolve1d(convolve1d(L_y**2,gauss1D_spatial,axis=0, mode= 'reflect'),gauss1D_spatial,axis = 1, mode= 'reflect'), gauss1D_time, axis = 2, mode= 'reflect')
    M33 = convolve1d(convolve1d(convolve1d(L_t**2,gauss1D_spatial,axis=0, mode= 'reflect'),gauss1D_spatial,axis = 1, mode= 'reflect'), gauss1D_time, axis = 2, mode= 'reflect')
    M12 = convolve1d(convolve1d(convolve1d(L_x*L_y,gauss1D_spatial,axis=0, mode= 'reflect'),gauss1D_spatial,axis = 1, mode= 'reflect'), gauss1D_time, axis = 2, mode= 'reflect')
    M13 = convolve1d(convolve1d(convolve1d(L_x*L_t,gauss1D_spatial,axis=0, mode= 'reflect'),gauss1D_spatial,axis = 1, mode= 'reflect'), gauss1D_time, axis = 2, mode= 'reflect')
    M23 = convolve1d(convolve1d(convolve1d(L_y*L_t,gauss1D_spatial,axis=0, mode= 'reflect'),gauss1D_spatial,axis = 1, mode= 'reflect'), gauss1D_time, axis = 2, mode= 'reflect')
    
    # Determinant of M matrix
    det_M = -M33*(M12**2) + 2*M12*M13*M23 - M22 * (M13**2) - M11*(M23**2) + M11*M22*M33
    
    # Trace of M matrix
    trace_M = M11 + M22 + M33
    
    H = det_M - k * (trace_M)**3                                    # Cornerness criterion H
    
    # 2.1.3
    voxels_scale = Interest_Points(H,sigma)
    
    return voxels_scale


# In[10]:


N = HarrisDetector(frames,2,1.5,1,0.003)


# ### 2.1.2  Gabor Detector

# In[11]:


def GaborDetector(frames,sigma,taf):

    # Gaussian kernel
    n = int(2*np.ceil(3*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, sigma)                        # Column vector
    
    gauss1D = list(chain.from_iterable(gauss1D))                     # Convert gaussian kernel to list
        
    # Image smoothing with spatial Gaussian
    smoothed_frames = convolve1d(frames,gauss1D,axis = 0, mode='reflect')            
    smoothed_frames = convolve1d(smoothed_frames,gauss1D,axis = 1, mode='reflect')
        
    t = np.linspace(int(-2*taf),int(2*taf),int(4*taf+1))             # Define time variable
    omega = 4 / taf                                                  # Frequency omega in terms of time scale taf

    h_ev = np.cos(2*pi*t*omega)*np.exp((-(t**2))/(2*(taf**2)))       # h_ev calculation
    h_odd = np.sin(2*pi*t*omega)*np.exp((-(t**2))/(2*(taf**2)))      # h_odd calculation
    
    h_ev = h_ev / np.linalg.norm(h_ev,ord=1)                         # Normalise h_ev with l1 norm
    h_odd = h_odd / np.linalg.norm(h_odd,ord=1)                      # Normalise h_odd with l1 norm
    
    H_ev = convolve1d(smoothed_frames,h_ev,axis = 2, mode='reflect')                 # H_ev calculation
    H_odd = convolve1d(smoothed_frames,h_odd,axis = 2, mode='reflect')               # H_odd calculation 
       
    H = H_ev**2 + H_odd**2                                           # H calculation

    # 2.1.3
    voxels_scale = Interest_Points(H,sigma)
    
    return voxels_scale


# In[12]:


N = GaborDetector(frames,2,1.5)


# ### 2.2 Χωρο-χρονικοί Ιστογραφικοί Περιγραφητές

# ### Περιγραφητής HOG

# In[13]:


def HOG(frames,InterestPoints):
    
################################################# 2.2.1 ####################################################################

    nbins = 9                                                   # Number of bins
    n = 2                                                       # Grid width
    m = 2                                                       # Grid height
    
    desc = np.zeros((InterestPoints.shape[0],nbins*n*m))        # Initialisate array of descriptors

    # Calculation of descriptor for every interest point 
    for i in range(InterestPoints.shape[0]):

        x = int(InterestPoints[i,0])                            # x coordinate
        y = int(InterestPoints[i,1])                            # y coordinate
        t = int(InterestPoints[i,2])                            # frame number
        sigma = int(InterestPoints[i,3])                        # scale 
        box = 4*sigma                                           # Defines local neighbourhood of interest point

        I = frames[:,:,t]                                       # Read the frame that contains the interest point
        
        kernel = [-1,0,1]                                       # kernel for derivatives calculation
        I_x = convolve1d(I,kernel,axis = 0, mode='reflect')     # Calculate Ix by convolving I with kernel in x-dimension
        I_y = convolve1d(I,kernel,axis = 1, mode='reflect')     # Calculate Iy by convolving I with kernel in y-dimension
        
        # Definition of bounding box 
        x_left = max(x - int(box),0)
        x_right = min(x + int(box),I.shape[1])  
        y_up = max(y - int(box),0)
        y_down = min(y+int(box),I.shape[0]) 
        
        # Cropping on the bounding box    
        I_cropped_x = I_x[y_up:(y_down+1),x_left:(x_right+1)]
        I_cropped_y = I_y[y_up:(y_down+1),x_left:(x_right+1)]
        
        # Descriptor calculation
        desc[i,:]= p.orientation_histogram(I_cropped_x,I_cropped_y,nbins,np.array([n,m]))
        
        
    return desc


# ### Περιγραφητής HOF

# In[14]:


def HOF(frames,InterestPoints):
    
################################################# 2.2.1 ####################################################################

    nbins = 9                                                   # Number of bins
    n = 2                                                       # Grid width
    m = 2                                                       # Grid height
        
    desc = np.zeros((InterestPoints.shape[0],nbins*n*m))        # Initialisate array of descriptors
        
    for i in range(InterestPoints.shape[0]):
            
        x = int(InterestPoints[i,0])                            # x coordinate
        y = int(InterestPoints[i,1])                            # y coordinate
        t = int(InterestPoints[i,2])                            # frame number
        sigma = int(InterestPoints[i,3])                        # scale 
        box = 4*sigma                                           # Defines local neighbourhood of interest point
        
        Iprev = frames[:,:,t]                                   # Read the frame that contains the interest point
        Inext = frames[:,:,t+1]                                 # Read the next frame
                 
        d_x,d_y = lk(Iprev,Inext,3,0.07,0,0)                    # Calculate optical flow with LK 
    
        # Definition of bounding box 
        x_left = max(x - int(box),0)
        x_right = min(x + int(box),Iprev.shape[1])  
        y_up = max(y - int(box),0)
        y_down = min(y+int(box),Iprev.shape[0])
        
        # Cropping on the bounding box    
        d_x_cropped = d_x[y_up:(y_down+1),x_left:(x_right+1)]
        d_y_cropped = d_y[y_up:(y_down+1),x_left:(x_right+1)] 
        
################################################# 2.2.2 ####################################################################
        
        # Descriptor calculation
        desc[i,:]= p.orientation_histogram(d_x_cropped,d_y_cropped,nbins,np.array([n,m]))
                        
    return desc


# ### Περιγραφητής HOG/HOF

# In[15]:


def HOG_HOF(frames,InterestPoints):
    hog = HOG(frames,InterestPoints)
    hof = HOF(frames,InterestPoints)
    hog_hof = np.concatenate((hog, hof), axis=1)
    return hog_hof


# ### Υλοποίηση περιγραφητών

# In[16]:


def CreateDescriptors(frames,InterestPoints,descriptor):
    if descriptor=="HOG":
        desc = HOG(frames,InterestPoints)
        return desc
    elif descriptor=="HOF":
        desc = HOF(frames,InterestPoints)
        return desc
    elif descriptor=="HOG/HOF":
        desc = HOG_HOF(frames,InterestPoints)
        return desc
    else: print("No such descriptor!We are sorry!")


# In[3]:


def myHistogram(Fx,Fy): # Για σ = 4
    mag, angle = cv2.cartToPolar(Fx, Fy, angleInDegrees=True)
    angle = angle % 180
    bins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    histogram = np.zeros((49,9))
    cell_counter = 0
    for i in range(0,resized_img.shape[0],5):
        for j in range(0,resized_img.shape[1],5):
            hist = np.zeros((1,9))
            if i!=30 and j!=30:
                gradient_magnitude = mag[i:(i+5),j:(j+5)]
                gradient_direction = angle[i:(i+5),j:(j+5)]
            elif i!=30 and j==30:
                gradient_magnitude = mag[i:(i+5),29:34]
                gradient_direction = angle[i:(i+5),29:34]
            elif i==30 and j!=30:
                gradient_magnitude = mag[29:34,j:(j+5)]
                gradient_direction = angle[29:34,j:(j+5)]
            else:
                gradient_magnitude = mag[29:34,29:34]
                gradient_direction = angle[29:34,29:34]
            for i in range(0,5):
                for j in range(0,5):
                        greater_bin = gradient_direction[i][j] // 20
                        portion = gradient_direction[i][j] % 20
                        hist[0,greater_bin + 1] = (portion/20) * gradient_magnitude[i][j]
                        hist[0,greater_bin] = (20-portion/20) * gradient_magnitude[i][j]
                        hist = hist / np.linalg.norm(hist)
            histogram[cell_counter,:] = hist
            cell_counter = cell_counter + 1    
    return histogram.flatten()


# ### 2.3: Κατασκευή Bag of Visual Words και χρήση Support Vector Machines για την ταξινόμηση δράσεων

# In[97]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np

train = open("cv20_lab2_part2_material/data_train/train_videos.txt", "r")
test = open("cv20_lab2_part2_material/data_test/test_videos.txt", "r")
training_videos = train.readlines()
testing_videos = test.readlines()
train_labels, test_labels = [], []
desc_train, desc_test = [], []
acc = 0

for i in range(5):
    for video in training_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        train_video = train_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        InterestPoints = p2.HarrisDetector(train_video,4,1.5,2,0.005)
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG")]


    for video in testing_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        test_video = test_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        InterestPoints = p2.HarrisDetector(test_video,4,1.5,2,0.005)
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=100)
    accuracy, pred = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy
final_acc = acc/5
print('Accuracy for HarrisDetector with HOG descriptors: {:.3f}%'.format(100.0*final_acc))


# In[98]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np

train = open("cv20_lab2_part2_material/data_train/train_videos.txt", "r")
test = open("cv20_lab2_part2_material/data_test/test_videos.txt", "r")
training_videos = train.readlines()
testing_videos = test.readlines()
train_labels, test_labels = [], []
desc_train, desc_test = [], []
acc = 0

for i in range(5):
    for video in training_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        train_video = train_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        InterestPoints = p2.HarrisDetector(train_video,4,1.5,2,0.005)
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOF")]


    for video in testing_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        test_video = test_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        InterestPoints = p2.HarrisDetector(test_video,4,1.5,2,0.005)
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=100)
    accuracy, pred = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy
final_acc = acc/5
print('Accuracy for HarrisDetector with HOF descriptors: {:.3f}%'.format(100.0*final_acc))


# In[99]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np

train = open("cv20_lab2_part2_material/data_train/train_videos.txt", "r")
test = open("cv20_lab2_part2_material/data_test/test_videos.txt", "r")
training_videos = train.readlines()
testing_videos = test.readlines()
train_labels, test_labels = [], []
desc_train, desc_test = [], []
acc = 0

for i in range(5):
    for video in training_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        train_video = train_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        InterestPoints = p2.HarrisDetector(train_video,4,1.5,2,0.005)
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG/HOF")]


    for video in testing_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        test_video = test_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        InterestPoints = p2.HarrisDetector(test_video,4,1.5,2,0.005)
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG/HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=100)
    accuracy, pred = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy
final_acc = acc/5
print('Accuracy for HarrisDetector with HOG/HOF descriptors: {:.3f}%'.format(100.0*final_acc))


# In[100]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np

train = open("cv20_lab2_part2_material/data_train/train_videos.txt", "r")
test = open("cv20_lab2_part2_material/data_test/test_videos.txt", "r")
training_videos = train.readlines()
testing_videos = test.readlines()
train_labels, test_labels = [], []
desc_train, desc_test = [], []
acc = 0
for i in range(5):
    for video in training_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        train_video = train_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        InterestPoints = p2.GaborDetector(train_video,3,1.5)
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG")]


    for video in testing_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        test_video = test_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        InterestPoints = p2.GaborDetector(test_video,3,1.5)
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=70)
    accuracy, pred = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy
final_acc = acc/5
print('Accuracy for GaborDetector with HOG descriptors: {:.3f}%'.format(100.0*final_acc))


# In[101]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np

train = open("cv20_lab2_part2_material/data_train/train_videos.txt", "r")
test = open("cv20_lab2_part2_material/data_test/test_videos.txt", "r")
training_videos = train.readlines()
testing_videos = test.readlines()
train_labels, test_labels = [], []
desc_train, desc_test = [], []
acc = 0

for i in range(5):
    for video in training_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        train_video = train_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        InterestPoints = p2.GaborDetector(train_video,3,1.5)
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOF")]


    for video in testing_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        test_video = test_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        InterestPoints = p2.GaborDetector(test_video,3,1.5)
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=70)
    accuracy, pred = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy
final_acc = acc/5
print('Accuracy for GaborDetector with HOF descriptors: {:.3f}%'.format(100.0*final_acc))


# In[102]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np

train = open("cv20_lab2_part2_material/data_train/train_videos.txt", "r")
test = open("cv20_lab2_part2_material/data_test/test_videos.txt", "r")
training_videos = train.readlines()
testing_videos = test.readlines()
train_labels, test_labels = [], []
desc_train, desc_test = [], []
acc = 0

for i in range(5):
    for video in training_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        train_video = train_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        InterestPoints = p2.GaborDetector(train_video,4,1.5)
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG/HOF")]


    for video in testing_videos:
        if video == '\n': break
        action = video.rstrip().split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video.rstrip()),200,0)
        test_video = test_video.astype(np.float)/255
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        InterestPoints = p2.GaborDetector(test_video,4,1.5)
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG/HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=70)
    accuracy, pred = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy
final_acc = acc/5
print('Accuracy for GaborDetector with HOG/HOF descriptors: {:.3f}%'.format(100.0*final_acc))


# In[103]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np
import random

videos = open("cv20_lab2_part2_material/videos.txt", "r")
running_videos = open("cv20_lab2_part2_material/running/running.txt", "r")
boxing_videos = open("cv20_lab2_part2_material/boxing/boxing.txt", "r")
walking_videos = open("cv20_lab2_part2_material/walking/walking.txt", "r")

running_video_list, boxing_video_list, walking_video_list, video_list = [], [], [], []
acc = 0

for video in videos.readlines():
    if video == '\n' : break
    video_list = video_list + [video.rstrip()]

for running, boxing, walking in zip(running_videos.readlines(),boxing_videos.readlines(),walking_videos.readlines()):
    running_video_list = running_video_list + [running.rstrip()]
    boxing_video_list = boxing_video_list + [boxing.rstrip()]
    walking_video_list = walking_video_list + [walking.rstrip()]
for i in range(5):
    running_training_videos = random.sample(running_video_list,12)
    boxing_training_videos = random.sample(boxing_video_list,12)
    walking_training_videos = random.sample(walking_video_list,12)

    training_videos = running_training_videos + boxing_training_videos + walking_training_videos
    testing_videos = list(set(video_list)-set(training_videos))

    train_labels, test_labels = [], []
    desc_train, desc_test = [], []

    for video in training_videos:
        action = video.split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        train_video = train_video.astype(np.float)/255
        InterestPoints = p2.HarrisDetector(train_video,4,1.5,2,0.005)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG")]

    for video in testing_videos:
        action = video.split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        test_video = test_video.astype(np.float)/255
        InterestPoints = p2.HarrisDetector(test_video,4,1.5,2,0.005)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=100)
    accuracy, _ = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy

final_accuracy = acc/5
print('Accuracy for HarrisDetector with HOG descriptors: {:.3f}%'.format(100.0*final_accuracy))


# In[2]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np
import random

videos = open("cv20_lab2_part2_material/videos.txt", "r")
running_videos = open("cv20_lab2_part2_material/running/running.txt", "r")
boxing_videos = open("cv20_lab2_part2_material/boxing/boxing.txt", "r")
walking_videos = open("cv20_lab2_part2_material/walking/walking.txt", "r")

running_video_list, boxing_video_list, walking_video_list, video_list = [], [], [], []
acc = 0

for video in videos.readlines():
    if video == '\n' : break
    video_list = video_list + [video.rstrip()]

for running, boxing, walking in zip(running_videos.readlines(),boxing_videos.readlines(),walking_videos.readlines()):
    running_video_list = running_video_list + [running.rstrip()]
    boxing_video_list = boxing_video_list + [boxing.rstrip()]
    walking_video_list = walking_video_list + [walking.rstrip()]
for i in range(5):
    running_training_videos = random.sample(running_video_list,12)
    boxing_training_videos = random.sample(boxing_video_list,12)
    walking_training_videos = random.sample(walking_video_list,12)

    training_videos = running_training_videos + boxing_training_videos + walking_training_videos
    testing_videos = list(set(video_list)-set(training_videos))

    train_labels, test_labels = [], []
    desc_train, desc_test = [], []

    for video in training_videos:
        action = video.split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        train_video = train_video.astype(np.float)/255
        InterestPoints = p2.HarrisDetector(train_video,4,1.5,2,0.005)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOF")]

    for video in testing_videos:
        action = video.split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        test_video = test_video.astype(np.float)/255
        InterestPoints = p2.HarrisDetector(test_video,4,1.5,2,0.005)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=100)
    accuracy, _ = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy

final_accuracy = acc/5
print('Accuracy for HarrisDetector with HOF descriptors: {:.3f}%'.format(100.0*final_accuracy))


# In[105]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np
import random

videos = open("cv20_lab2_part2_material/videos.txt", "r")
running_videos = open("cv20_lab2_part2_material/running/running.txt", "r")
boxing_videos = open("cv20_lab2_part2_material/boxing/boxing.txt", "r")
walking_videos = open("cv20_lab2_part2_material/walking/walking.txt", "r")

running_video_list, boxing_video_list, walking_video_list, video_list = [], [], [], []
acc = 0

for video in videos.readlines():
    if video == '\n' : break
    video_list = video_list + [video.rstrip()]

for running, boxing, walking in zip(running_videos.readlines(),boxing_videos.readlines(),walking_videos.readlines()):
    running_video_list = running_video_list + [running.rstrip()]
    boxing_video_list = boxing_video_list + [boxing.rstrip()]
    walking_video_list = walking_video_list + [walking.rstrip()]
for i in range(5):
    running_training_videos = random.sample(running_video_list,12)
    boxing_training_videos = random.sample(boxing_video_list,12)
    walking_training_videos = random.sample(walking_video_list,12)

    training_videos = running_training_videos + boxing_training_videos + walking_training_videos
    testing_videos = list(set(video_list)-set(training_videos))

    train_labels, test_labels = [], []
    desc_train, desc_test = [], []

    for video in training_videos:
        action = video.split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        train_video = train_video.astype(np.float)/255
        InterestPoints = p2.HarrisDetector(train_video,4,1.5,2,0.005)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG/HOF")]

    for video in testing_videos:
        action = video.split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        test_video = test_video.astype(np.float)/255
        InterestPoints = p2.HarrisDetector(test_video,4,1.5,2,0.005)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG/HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=100)
    accuracy, _ = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy

final_accuracy = acc/5
print('Accuracy for HarrisDetector with HOG/HOF descriptors: {:.3f}%'.format(100.0*final_accuracy))


# In[106]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np
import random

videos = open("cv20_lab2_part2_material/videos.txt", "r")
running_videos = open("cv20_lab2_part2_material/running/running.txt", "r")
boxing_videos = open("cv20_lab2_part2_material/boxing/boxing.txt", "r")
walking_videos = open("cv20_lab2_part2_material/walking/walking.txt", "r")

running_video_list, boxing_video_list, walking_video_list, video_list = [], [], [], []
acc = 0

for video in videos.readlines():
    if video == '\n' : break
    video_list = video_list + [video.rstrip()]

for running, boxing, walking in zip(running_videos.readlines(),boxing_videos.readlines(),walking_videos.readlines()):
    running_video_list = running_video_list + [running.rstrip()]
    boxing_video_list = boxing_video_list + [boxing.rstrip()]
    walking_video_list = walking_video_list + [walking.rstrip()]
for i in range(5):
    running_training_videos = random.sample(running_video_list,12)
    boxing_training_videos = random.sample(boxing_video_list,12)
    walking_training_videos = random.sample(walking_video_list,12)

    training_videos = running_training_videos + boxing_training_videos + walking_training_videos
    testing_videos = list(set(video_list)-set(training_videos))

    train_labels, test_labels = [], []
    desc_train, desc_test = [], []

    for video in training_videos:
        action = video.split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        train_video = train_video.astype(np.float)/255
        InterestPoints = p2.GaborDetector(train_video,3,1.5)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG")]

    for video in testing_videos:
        action = video.split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        test_video = test_video.astype(np.float)/255
        InterestPoints = p2.GaborDetector(test_video,3,1.5)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=70)
    accuracy, _ = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy

final_accuracy = acc/5
print('Accuracy for GaborDetector with HOG descriptors: {:.3f}%'.format(100.0*final_accuracy))


# In[107]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np
import random

videos = open("cv20_lab2_part2_material/videos.txt", "r")
running_videos = open("cv20_lab2_part2_material/running/running.txt", "r")
boxing_videos = open("cv20_lab2_part2_material/boxing/boxing.txt", "r")
walking_videos = open("cv20_lab2_part2_material/walking/walking.txt", "r")

running_video_list, boxing_video_list, walking_video_list, video_list = [], [], [], []
acc = 0

for video in videos.readlines():
    if video == '\n' : break
    video_list = video_list + [video.rstrip()]

for running, boxing, walking in zip(running_videos.readlines(),boxing_videos.readlines(),walking_videos.readlines()):
    running_video_list = running_video_list + [running.rstrip()]
    boxing_video_list = boxing_video_list + [boxing.rstrip()]
    walking_video_list = walking_video_list + [walking.rstrip()]
for i in range(5):
    running_training_videos = random.sample(running_video_list,12)
    boxing_training_videos = random.sample(boxing_video_list,12)
    walking_training_videos = random.sample(walking_video_list,12)

    training_videos = running_training_videos + boxing_training_videos + walking_training_videos
    testing_videos = list(set(video_list)-set(training_videos))

    train_labels, test_labels = [], []
    desc_train, desc_test = [], []

    for video in training_videos:
        action = video.split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        train_video = train_video.astype(np.float)/255
        InterestPoints = p2.GaborDetector(train_video,3,1.5)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOF")]

    for video in testing_videos:
        action = video.split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        test_video = test_video.astype(np.float)/255
        InterestPoints = p2.GaborDetector(test_video,3,1.5)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=70)
    accuracy, _ = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy

final_accuracy = acc/5
print('Accuracy for GaborDetector with HOF descriptors: {:.3f}%'.format(100.0*final_accuracy))


# In[108]:


import cv20_lab2_2_utils as p1
import cv20_lab2_2_1_2 as p2
import numpy as np
import random

videos = open("cv20_lab2_part2_material/videos.txt", "r")
running_videos = open("cv20_lab2_part2_material/running/running.txt", "r")
boxing_videos = open("cv20_lab2_part2_material/boxing/boxing.txt", "r")
walking_videos = open("cv20_lab2_part2_material/walking/walking.txt", "r")

running_video_list, boxing_video_list, walking_video_list, video_list = [], [], [], []
acc = 0

for video in videos.readlines():
    if video == '\n' : break
    video_list = video_list + [video.rstrip()]

for running, boxing, walking in zip(running_videos.readlines(),boxing_videos.readlines(),walking_videos.readlines()):
    running_video_list = running_video_list + [running.rstrip()]
    boxing_video_list = boxing_video_list + [boxing.rstrip()]
    walking_video_list = walking_video_list + [walking.rstrip()]
for i in range(5):
    running_training_videos = random.sample(running_video_list,12)
    boxing_training_videos = random.sample(boxing_video_list,12)
    walking_training_videos = random.sample(walking_video_list,12)

    training_videos = running_training_videos + boxing_training_videos + walking_training_videos
    testing_videos = list(set(video_list)-set(training_videos))

    train_labels, test_labels = [], []
    desc_train, desc_test = [], []

    for video in training_videos:
        action = video.split("_")[1]
        train_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        train_video = train_video.astype(np.float)/255
        InterestPoints = p2.GaborDetector(train_video,3,1.5)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        train_labels = train_labels + [label]
        desc_train = desc_train + [p2.CreateDescriptors(train_video,InterestPoints,"HOG/HOF")]

    for video in testing_videos:
        action = video.split("_")[1]
        test_video = p1.read_video('cv20_lab2_part2_material/{folder_name}/{video_name}'.format(folder_name = action, video_name = video),200,0)
        test_video = test_video.astype(np.float)/255
        InterestPoints = p2.GaborDetector(test_video,3,1.5)
        if action=='running':
            label = 0
        elif action=='boxing':
            label = 1
        else:
            label = 2
        test_labels = test_labels + [label]
        desc_test = desc_test + [p2.CreateDescriptors(test_video,InterestPoints,"HOG/HOF")]

    bow_train, bow_test = p1.bag_of_words(desc_train, desc_test, num_centers=70)
    accuracy, _ = p1.svm_train_test(bow_train, train_labels, bow_test, test_labels)
    acc = acc + accuracy

final_accuracy = acc/5
print('Accuracy for GaborDetector with HOG/HOF descriptors: {:.3f}%'.format(100.0*final_accuracy))


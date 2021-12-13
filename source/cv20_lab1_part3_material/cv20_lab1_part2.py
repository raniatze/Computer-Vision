import numpy as np
import cv2
import math
from math import pi
import cv20_lab1_part1 as p1


def AngleDetect(image_gray,sigma,r,k,theta_corn):

##################################################### 2.1.1 ########################################################################

    gauss2D_sigma, _ = p1.CreateFilters(sigma)

    gauss2D_r, _ = p1.CreateFilters(r)

    Is = cv2.filter2D(image_gray, -1, gauss2D_sigma)                 # Smoothed image
    [Isx,Isy] = np.gradient(Is)                                      # Gradient of Is

    J1 = cv2.filter2D((Isx*Isx), -1, gauss2D_r)                      # J1 Calculation
    J2 = cv2.filter2D((Isx*Isy), -1, gauss2D_r)                      # J2 Calculation
    J3 = cv2.filter2D((Isy*Isy), -1, gauss2D_r)                      # J3 Calculation

####################################################### 2.1.2 ######################################################################

    lambda_pos = (1/2)*(J1+J3+np.sqrt((J1-J3)**2+4*(J2**2)))         # Positive eigenvalue
    lambda_neg = (1/2)*(J1+J3-np.sqrt((J1-J3)**2+4*(J2**2)))         # Negative eigenvalue

    #fig, ax = plt.subplots(1, 2, figsize=(15,15))
    #ax[0].set_title('Lambda positive for σ = {}'.format(round(sigma,4)))
    #ax[0].imshow(lambda_pos,cmap='gray')
    #ax[1].set_title('Lambda negative for σ = {}'.format(round(sigma,4)))
    #ax[1].imshow(lambda_neg,cmap='gray')

####################################################### 2.1.3 ######################################################################

    R = lambda_neg*lambda_pos - k * (lambda_neg + lambda_pos)**2      # Cornerness criterion R

    ns = np.ceil(3*sigma)*2+1
    B_sq = p1.disk_strel(ns)
    Cond1 = (R==cv2.dilate(R,B_sq))                                   # Condition 1
    Cond2 = (R>(theta_corn*R.max()))                                  # Condition 2 (Thresholding)

    Corners = Cond1 & Cond2                                           # Combine both conditions
    Coordinates = np.argwhere(Corners)                                # Corner coordinates calculations
    Coordinates = np.flip(Coordinates,1)                              # Reverse coordinates
    Scale = np.ones((Coordinates.shape[0],1))*sigma                   # 1 column array with current scale
    Coordinates_Scale = np.concatenate((Coordinates, Scale), axis=1)  # Concatenate coordinates with scale


    return Coordinates_Scale, Corners



def MultiScalarAngleDetect(image_gray,sigma,r,scale,N,k,theta_corn):

    LoG_normalised, ScalarCorners, Corners, sigma_ = [0] * N, [0] * N, [0] * N, [0] * N

##################################################### 2.2.1 ########################################################################

    for i in range(0,N):

        sigma_[i] = (scale**i)*sigma                                                     # Current sigma
        r_ = (scale**i)*r                                                                # Current r

##################################################### 2.2.2 ########################################################################

        # Laplacian-of-Gaussian
        n = int(2*np.ceil(3*sigma_[i])+1)
        x = np.linspace(-n/2,n/2,n)
        x, y = np.meshgrid(x,x)
        LoG = (1/(2*pi*sigma_[i]**4))*((1/sigma_[i]**2)*(x**2+y**2)-2*np.ones(x.shape)) * np.exp(-(1/(2*sigma_[i]**2))*(x**2+y**2))
        LoG_normalised[i] = cv2.filter2D(image_gray, -1, LoG)
        LoG_normalised[i] = sigma_[i]**2 * np.absolute(LoG_normalised[i])                 # Normalised LoG


        ScalarCorners[i], Corners[i] = AngleDetect(image_gray,sigma_[i],r_,k,theta_corn)  # One scale corner detection

    # Reject certain points
    for i in range(0,N):

        if N==1:
            condition = np.ones(Corners[i].shape).astype(np.uint8)
        elif i==0:
            condition = (LoG_normalised[i] > LoG_normalised[i+1])
        elif i==N-1:
            condition = (LoG_normalised[i] > LoG_normalised[i-1])
        else:
            condition = (LoG_normalised[i] > LoG_normalised[i+1]) & (LoG_normalised[i] > LoG_normalised[i-1])

        Coordinates = np.argwhere(Corners[i] & condition)        # MultiScalarCorner coordinates calculation
        Coordinates = np.flip(Coordinates,1)                     # Reverse coordinates
        Scale = np.ones((Coordinates.shape[0],1))*sigma_[i]      # 1 column array with current scale
        current = np.concatenate((Coordinates, Scale), axis=1)   # Concatenate coordinates with scale

        if N==1:
            MultiScalarCorners = current
        elif i==0:
            prev = current
        else:
            MultiScalarCorners = np.concatenate((prev,current), axis=0)
            prev = MultiScalarCorners

    return MultiScalarCorners


def BlobsDetect(image_gray,sigma,theta_corn):

##################################################### 2.3.1 ########################################################################

    gauss2D, _ = p1.CreateFilters(sigma)

    Is = cv2.filter2D(image_gray, -1, gauss2D)                         # Smoothed image
    [Lx,Ly] = np.gradient(Is)                                          # Gradient of Is
    [Lxx,_] = np.gradient(Lx)                                          # Gradient of Isx
    [Lxy,Lyy] = np.gradient(Ly)                                        # Gradient of Isy
    R = Lxx*Lyy - Lxy**2                                               # R criterion

##################################################### 2.3.2 ########################################################################

    ns = np.ceil(3*sigma)*2+1
    B_sq = p1.disk_strel(ns)
    Cond1 = (R==cv2.dilate(R,B_sq))                                    # Condition 1
    Cond2 = (R>(theta_corn*R.max()))                                   # Condition 2 (Thresholding)

    Blobs = Cond1 & Cond2                                              # Combine conditions
    Coordinates = np.argwhere(Blobs)                                   # Blob coordinates
    Coordinates = np.flip(Coordinates,1)                               # Reverse coordinates
    scale = (np.ones((Coordinates.shape[0],1))*sigma)                  # 1 column array with current scale
    Coordinates_Scale = np.concatenate((Coordinates, scale), axis=1)   # Concatenate coordinates with scales

    return Coordinates_Scale, Blobs

def MultiScalarBlobsDetect(image_gray,sigma,scale,N,theta_corn):

    LoG_normalised, ScalarBlobs, Blobs, sigma_ = [0] * N, [0] * N, [0] * N, [0] * N

    for i in range(0,N):

        sigma_[i] = (scale**i)*sigma                                             # Current sigma

        # Laplacian-of-Gaussian
        n = int(2*np.ceil(3*sigma_[i])+1)
        x = np.linspace(-n/2,n/2,n)
        x, y = np.meshgrid(x,x)
        LoG = (1/(2*pi*sigma_[i]**4))*((1/sigma_[i]**2)*(x**2+y**2)-2*np.ones(x.shape)) * np.exp(-(1/(2*sigma_[i]**2))*(x**2+y**2))
        LoG_normalised[i] = cv2.filter2D(image_gray, -1, LoG)
        LoG_normalised[i] = sigma_[i]**2 * np.absolute(LoG_normalised[i])        # Normalised LoG

        ScalarBlobs[i], Blobs[i] = BlobsDetect(image_gray,sigma_[i],theta_corn)  # One Scalar blob detection

    # Reject certain points
    for i in range(0,N):

        if N==1:
            condition = np.ones(Blobs[i].shape).astype(np.uint8)
        elif i==0:
            condition = (LoG_normalised[i] > LoG_normalised[i+1])
        elif i==N-1:
            condition = (LoG_normalised[i] > LoG_normalised[i-1])
        else:
            condition = (LoG_normalised[i] > LoG_normalised[i+1]) & (LoG_normalised[i] > LoG_normalised[i-1])

        Coordinates = np.argwhere(Blobs[i] & condition)                         # MultiScalarBlob coordinates calculation
        Coordinates = np.flip(Coordinates,1)                                    # Reverse coordinates
        Scale = np.ones((Coordinates.shape[0],1))*sigma_[i]                     # 1 column array with current scale
        current = np.concatenate((Coordinates, Scale), axis=1)                  # Concatenate coordinates with scale

        if N==1:
            MultiScalarBlobs = current
        elif i==0:
            prev = current
        else:
            MultiScalarBlobs = np.concatenate((prev,current), axis=0)
            prev = MultiScalarBlobs

    return MultiScalarBlobs


def shiftedImage(image,shiftX,shiftY,offsetx, offsety):

    S_A = np.roll(image,int(shiftY+offsety),axis=0)
    S_A = np.roll(S_A, int(shiftX+offsetx),axis=1)

    S_B = np.roll(image,int(shiftY+offsety),axis=0)
    S_B = np.roll(S_B,int(-shiftX+offsetx),axis=1)

    S_C = np.roll(image,int(-shiftY+offsety),axis=0)
    S_C = np.roll(S_C,int(-shiftX+offsetx),axis=1)

    S_D = np.roll(image,int(-shiftY+offsety),axis=0)
    S_D = np.roll(S_D, int(shiftX+offsetx),axis=1)

    return S_A , S_B, S_C , S_D



def BoxFilter(image,sigma,theta_corn):

##################################################### 2.5.1 ########################################################################

    n = 2*np.ceil(3*sigma)+1
    pad = np.floor(n/2)+1                                    # padding factor
    image_pad = np.pad(image,int(pad),'edge')                # Image padding


    for i in range(image_pad.ndim):                          # Integral image calculation (after padding)
        image_pad = image_pad.cumsum(axis=i)

##################################################### 2.5.2 ########################################################################

    a = 2*np.floor(n/6) + 1                                  # Box filter dimension parameter
    b = 4*np.floor(n/6) + 1                                  # Box filter dimension parameter

    shiftA = (a-1)/2                                         # Shifting parameter
    shiftB = (b-1)/2                                         # Shifting parameter

    if(np.ceil((n-2*a)/3) % 2 == 1):                         # r calculation for offset parameter
        r = np.ceil((n-2*a)/3)
    else:
        r = np.floor((n-2*a)/3)


    # Lxx calculation by using integral image
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftA,shiftB,0,0)
    Lxx = -3*(S_C+S_A-S_B-S_D)
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftA+a,shiftB,0,0)
    Lxx = S_C+S_A-S_B-S_D + Lxx

    # Lyy calculation by using integral image
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftB,shiftA,0,0)
    Lyy = -3*(S_C+S_A-S_B-S_D)
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftB,shiftA+a,0,0)
    Lyy = S_C+S_A-S_B-S_D + Lyy


    # Lxy calculation by using integral image
    offsety = (r-1)/2 + shiftA
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftA,shiftA,-offsety, offsety)
    Lxy = -(S_C+S_A-S_B-S_D)
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftA,shiftA, offsety, offsety)
    Lxy = (S_C+S_A-S_B-S_D) + Lxy
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftA,shiftA, offsety,-offsety)
    Lxy = -(S_C+S_A-S_B-S_D) + Lxy
    [S_A,S_B,S_C,S_D] = shiftedImage(image_pad,shiftA,shiftA,-offsety,-offsety)
    Lxy = (S_C+S_A-S_B-S_D) + Lxy

##################################################### 2.5.3 ########################################################################

    R = Lxx*Lyy - (0.9*Lxy)**2                                               # R criterion
    R = R[int(pad):R.shape[0]-int(pad),int(pad):R.shape[1]-int(pad)]         # unpadding
    B_sq = p1.disk_strel(n)
    Cond1 = (R==cv2.dilate(R,B_sq))                                          # Condition 1
    Cond2 = (R>(theta_corn*R.max()))                                         # Condition 2
    interest_points = Cond1 & Cond2                                          # Combine conditions
    Coordinates = np.argwhere(interest_points)                               # Coordinates calculation
    Coordinates = np.flip(Coordinates,1)                                     # Reverse coordinates
    Scale = np.ones((Coordinates.shape[0],1))*sigma                          # 1 column array with scale
    Coordinates_Scale = np.concatenate((Coordinates, Scale), axis=1)         # Concatenate coordinates with scale

    return Coordinates_Scale, interest_points, R


def MultiScalarBoxFilter(image_gray,sigma,scale,N,theta_corn):

    LoG_normalised, ScalarBoxFilters, BoxFilters, sigma_ = [0] * N, [0] * N, [0] * N, [0] * N

    for i in range(0,N):

        sigma_[i] = (scale**i)*sigma                                               # Current sigma

        # Laplacian-of-Gaussian
        n = int(2*np.ceil(3*sigma_[i])+1)
        x = np.linspace(-n/2,n/2,n)
        x, y = np.meshgrid(x,x)
        LoG = (1/(2*pi*sigma_[i]**4))*((1/sigma_[i]**2)*(x**2+y**2)-2*np.ones(x.shape)) * np.exp(-(1/(2*sigma_[i]**2))*(x**2+y**2))
        LoG_normalised[i] = cv2.filter2D(image_gray, -1, LoG)
        LoG_normalised[i] = sigma_[i]**2 * np.absolute(LoG_normalised[i])          # Normalised LoG

        ScalarBoxFilters[i], BoxFilters[i], _  = BoxFilter(image_gray,sigma_[i],theta_corn)

    for i in range(0,N):

        if N==1:
            condition = np.ones(BoxFilters[i].shape).astype(np.uint8)
        elif i==0:
            condition = (LoG_normalised[i] > LoG_normalised[i+1])
        elif i==N-1:
            condition = (LoG_normalised[i] > LoG_normalised[i-1])
        else:
            condition = (LoG_normalised[i] > LoG_normalised[i+1]) & (LoG_normalised[i] > LoG_normalised[i-1])

        Coordinates = np.argwhere(BoxFilters[i] & condition)                      # Coordinates calculation
        Coordinates = np.flip(Coordinates,1)                                      # Reverse coordinates
        Scale = np.ones((Coordinates.shape[0],1))*sigma_[i]                       # 1 column array with current scale
        current = np.concatenate((Coordinates, Scale), axis=1)                    # Concatenate coordinates with scale

        if N==1:
            MultiScalarBoxFilters = current
        elif i==0:
            prev = current
        else:
            MultiScalarBoxFilters = np.concatenate((prev,current), axis=0)
            prev = MultiScalarBoxFilters

    return MultiScalarBoxFilters

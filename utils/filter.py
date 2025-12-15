## Horizontal and Vertical Shift of Contrast Sensitivity Function
# HShiftList is horizontal shift. The smaller the ratio is, the more blurry the image would be.
# VShiftList is vertical shift. The smaller the ratio is, the more low contrast the image would be. 
# Refer to Xiong et al., 2021 Fontiers in Neuroscience Table for corresponding acuity (logMAR) and contrast sensitivity (logCS) levels


import cv2
import math
import numpy as np
# import matplotlib.pyplot as plt
import os
from copy import deepcopy
import glob
import re
import io
from PIL import Image

## iPhone 15 pro max: 6.7 inch, 2796 x 1290 pixels, 460 ppi
## iPhone 15 pro : 6.1 inch, 2556 x 1179 pixels, 460 ppi
def add_filter(img,HShift,VShift,screen_reso = (2796,1290), screen_size = 13.3, camera = True, white_balance=False, resize=True):
    """ Add low vision filter to the input image
    Args:
        img (numpy array): img is the input image in numpy array format.
        HShift (float): Horizontal shift value.
        VShift (float): Vertical shift value.
        screen_reso (tuple, optional): Screen resolution in pixels. Defaults to (2796,1290).
        screen_size (float, optional): Screen size in inches. Defaults to 13.3.
        camera (bool, optional): Whether the image is taken by a camera. Defaults to True.
        white_balance (bool, optional): Whether to apply white balance. Defaults to False.
        resize (bool, optional): Whether to resize the image to the screen resolution. Defaults to True.
    Returns:
        numpy array: The filtered image.
    """
    
    charIm = deepcopy(img)
    thisHShift = HShift
    thisVShift = VShift
    original_v, original_h = charIm.shape[:2]
    h=charIm.shape[1] # horizontal pixel number of the image
    v=charIm.shape[0] # vertical pixel number of the image
    aspect_ratio = h/v # aspect ratio of the image
    if h > v:
        h = screen_reso[0]
        v = int(h/aspect_ratio)
    else:
        v = screen_reso[1]
        h = int(v*aspect_ratio)
    charIm = cv2.resize(charIm,(h,v), interpolation = cv2.INTER_AREA)
    
    
    
    # if image not taken by a camera therefore viewing angle depend on the viewing distance
    if not camera:
        # Calculate Viewing Angle
        # PPI = sqrt((horizontal reso/width in inches)^2 + (vertical reso/height in inches)^2)
        # PPcm = PPI/2.54
        PPI = np.sqrt(screen_reso[0]**2 + screen_reso[1]**2)/screen_size
        ppcm = PPI/2.54
        PsysicalWidth = h/ppcm # physical width/height of the image on the screen (cm)
        PsysicalHeight = v/ppcm # physical width/height of the image on the screen (cm)
        distance=40 # Viewing distance in cm
        vh = 2*math.atan((PsysicalWidth)/(2*distance))*(180/math.pi) #horizontal visual angle of the image at the specified viewing distance
        vv = 2*math.atan((PsysicalHeight)/(2*distance))*(180/math.pi) #vertival visual angle of the image at the specified viewing distance
        imgSize = vh*vv #visual angle of the entire image at the specified viewing distance
    else:
        if v > h:
            vv = 71
            vh = 56
        else:  
            vh = 71 # iPhone main camera horizontal field of view in degrees
            vv = 56 # iPhone main camera vertical field of view in degrees
        imgSize = vh*vv # visual angle of the entire image 
        
    
    fx = np.arange(start=-h/2, stop=h/2, step=1)
    fx = fx/vh
    fy = np.arange(start=-v/2, stop=v/2, step=1)
    fy = fy/vv
    [ux,uy] = np.meshgrid(fx,fy)
    finalImg = np.zeros_like(charIm)
    for j in range(3): # three color channels or only luminance channel
        thisimage = charIm[:,:,j].astype(np.float64)
        meanLum = np.round(np.mean(thisimage),4)
        ## Generate blur
        ## Horizontal shift
        sSF0 = np.round(np.sqrt(np.round(ux**2+uy**2+.0001,4)),4)
        CSF0 = np.round((5200*np.exp(-.0016* (100/meanLum+1)**.08 * sSF0**2))/np.sqrt((0.64*sSF0**2+144/imgSize+1) * (1./(1-np.exp(-.02*sSF0**2))+63/(meanLum**.83))),4)

        sSF = thisHShift*np.round(np.sqrt(ux**2+uy**2+.0001),4)
        if white_balance:
            # Vertical Shift
            for ii in range(thisimage.shape[0]):
                for jj in range(thisimage.shape[1]):
                    if thisimage[ii,jj] !=255:
                        thisimage[ii,jj] = np.round(255-np.round((255-thisimage[ii,jj])*thisVShift,4))
            CSF = np.round((5200*np.exp(-.0016*(100/meanLum+1)**.08*sSF**2))/np.sqrt((0.64*sSF**2+144/imgSize+1) * (1./(1-np.exp(-.02*sSF**2))+63/(meanLum**.83))),4)
        else:
            CSF = thisVShift*np.round((5200*np.exp(-.0016*(100/meanLum+1)**.08*sSF**2))/np.sqrt((0.64*sSF**2+144/imgSize+1) * (1./(1-np.exp(-.02*sSF**2))+63/(meanLum**.83))),4)


        nCSF = np.fft.fftshift(np.round(CSF/CSF0,4))
        maxValue = 1
        nCSF = np.clip(nCSF,None,maxValue) #replace maximun to 1
        nCSF[0,0] = 1
        
        Y = np.fft.fft2(thisimage)
        filtImg = np.real(np.fft.ifft2(np.round(nCSF*Y,4)))
        
        ## put the three channels together
        finalImg[:,:,j] = np.clip(np.round(filtImg),0,255)

    if resize:
        finalImg = cv2.resize(finalImg,(original_h,original_v), interpolation = cv2.INTER_AREA)
    return finalImg



def add_filter_to_folder(imgFolder,expFolder,HShiftList, VShiftList, filters, camera_flag = True, white_flag = False):
    """ Add low vision filter to a folder of images

    Args:
        imgFolder (str): imgFolder is the directory where the input images are stored or the single input image.
        expFolder (str): expFolder is the directory where the output images will be saved.
        HShiftList (list): HShiftList is horizontal shift. The smaller the ratio is, the more blurry the image would be.
        VShiftList (list): VShiftList is vertical shift. The smaller the ratio is, the more low contrast the image would be. 
        filters (list, optional): filters used to add specific low vision effects to the input image. Defaults to [1].
        camera_flag (bool, optional): Flag to whether use iPhone Wide Camera or not. Defaults to True.
        white_flag (bool, optional): If set white as mean lum. Defaults to False.
    """
    if os.path.isdir(imgFolder):
        imgList = glob.glob(os.path.join(imgFolder,'*.*'))
    else:
        imgList = [imgFolder]
    if not os.path.exists(expFolder):
        os.makedirs(expFolder)
    for filter in filters:
        a = HShiftList[filter-1] # horizontal shift
        b = VShiftList[filter-1] # vertical shift
        for imgName in imgList:
            name = os.path.basename(imgName)
            outputImg =  os.path.join(expFolder,name)
            
            os.makedirs(os.path.dirname(outputImg), exist_ok=True)
            src_image = cv2.imread(imgName)
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB) # convert BGR to RGB

            thisHShift = np.round(1/a,4)
            thisVShift = b 
            finalImg = add_filter(src_image,thisHShift,thisVShift,camera=camera_flag,white_balance=white_flag, screen_reso = (1920,1200), screen_size = 16)
            ## Save Src and Filtered Images

            # Convert the NumPy array to a PIL image
            image = Image.fromarray(finalImg)
            image.save(outputImg, format='JPEG')
            



            

if __name__ == '__main__':
    HShiftList = [1.000, 0.288, 0.157, 0.086, 0.048, 0.027,
        0.250, 0.134, 0.072, 0.039, 0.022,
        0.267, 0.144, 0.078, 0.043, 0.024,
        0.314, 0.172, 0.096, 0.055, 0.032,
        0.345, 0.193, 0.110, 0.064, 0.038,
        0.439, 0.256, 0.154, 0.033, 0.018,
        0.125, 0.063, 0.031, 0.016, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000]

    VShiftList = [1.000, 0.288, 0.157, 0.086, 0.048, 0.027,
        1.000, 0.534, 0.288, 0.157, 0.086,
        0.534, 0.288, 0.157, 0.086, 0.048,
        0.157, 0.086, 0.048, 0.027, 0.016,
        0.086, 0.048, 0.027, 0.016, 0.010,
        0.027, 0.016, 0.010, 0.534, 0.288,
        1.000, 1.000, 1.000, 1.000, 0.355,
        0.178, 0.089, 0.045, 0.022, 0.011]
    
    filters=[2,3,4,5,6,7,32,33,34,35,36,38,39,40,41]
    filter_no_to_new = {1: 16, 2: 10, 3: 8, 4: 6, 5: 4, 6: 2, 7: 9, 
                        32: 7, 33: 5, 34: 3, 35: 1, 36: 15, 38: 14,
                        39: 13, 40: 12, 41: 11}
    camera_flag = False
    white_flag = False

    imgFolder = '/cis/home/qgao14/my_documents/VIOCR_infer_models/data/totaltext/16' #'imgs/IMG_5172.png' # can be directory or single image file
    for filter in filters:
        print(f'Processing filter {filter} ...')
        new_number = filter_no_to_new[filter]
        expFolder = f'/cis/home/qgao14/my_documents/VIOCR_infer_models/data/totaltext/{new_number}' # directory to save the output images

        add_filter_to_folder(imgFolder,expFolder, HShiftList, VShiftList, [filter], camera_flag = camera_flag, white_flag = white_flag)
  
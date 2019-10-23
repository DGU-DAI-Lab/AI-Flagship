import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model
import numpy as np
from PIL import Image, ImageTk, ImageFilter  
import tkinter as tk
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
from scipy.misc import imresize, imsave
from scipy import signal
import math

THRESHOULD_SM1 = 80
THRESHOULD_SM2 = 78

im = None
tk_img = None
tk_img1 = None
tk_img2 = None

# 파일 메뉴에서 “열기”를 선택하였을 때 호출되는 함수
def open():
    global im, tk_img
    fname = fd.askopenfilename()
    im = Image.open(fname)
    tk_img = ImageTk.PhotoImage(im)
    canvas.create_image(100, 100, image=tk_img)
    window.update()

# 파일 메뉴에서 “종료”를 선택하였을 때 호출되는 함수
def quit():
    window.quit()

# 영사처리 메뉴에서 “열기”를 선택하였을 때 호출되는 함수

def edge_detect(InputData,mask1,mask2):
    sobel1 = signal.convolve2d(InputData,mask1)
    sobel1 = sobel1[1:np.size(sobel1,0)-1,1:np.size(sobel1,1)-1]
    sobel2 = signal.convolve2d(InputData,mask2)
    sobel2 = sobel2[1:np.size(sobel2,0)-1,1:np.size(sobel2,1)-1]
    s = np.sqrt(sobel1**2 + sobel2**2)
    s = s*255/np.max(s)
    return s
    
def normal(InputData):
    m=np.zeros([InputData.shape[0],InputData.shape[1]])
    min_1 = np.min(InputData, axis = None)
    max_1=np.max(InputData, axis = None)
    for x in range(0,InputData.shape[0],1):
        for y in range(0,InputData.shape[1],1): 
            m[x,y] = (InputData[x,y]-min_1)*255/(max_1-min_1)
    
    return m
    
def pyramid(InputImage):
        
    Gaussian = np.array([[2,13,2],[13,40,13],[2,13,2]])
    totalbii = np.zeros([8,2])
    bi = np.uint8(InputImage)
    width = bi.shape[0]
    height = bi.shape[1]
    sub3 = np.zeros([128, 128*8])
    sub3[:,0:128] = InputImage
    for x in range (0,7,1):
        width = width//2
        height = height//2
        sub1 = imresize(bi, (width,height))
        sub1 = signal.convolve2d(sub1, Gaussian)
        sub1 = sub1[1:np.size(sub1,0)-1,1:np.size(sub1,1)-1]
        sub1 = normal(sub1)
        bi = sub1
        bi1 = sub1
        totalbi = np.array([[bi.shape[0],bi.shape[1]]])
        totalbii[0:1,:]=np.array([[InputImage.shape[0],InputImage.shape[1]]])
        totalbii[x+1:x+2,:] = totalbi
        width1 = bi1.shape[0]
        height1 = bi1.shape[1] 
        for y in range(0,x+1,1):         
            width1=width1*2
            height1=height1*2
            sub2 = imresize(bi1,(width1, height1),interp='bilinear')
            Q = (np.size(sub2,axis=0) != totalbii[x-y][0] or np.size(sub2,axis=1) != 1)
            W = (np.mod(np.size(sub2,axis=0)-totalbii[x-y][0], 2) == 0)
            E = (np.mod(np.size(sub2,axis=1)-totalbii[x-y][0], 2) == 0)
            if Q and W and E:
                #if ((sub2.shape[0] != totalbi[x+1-y,0] or sub2.shape[1] != 1) and (np.mod(sub2.shape[0]-totalbi[x+1-y,0],1)==0) and (np.mod(sub2.shape[1]-totalbi[x+1-y,0],1)==0):
                dif1 = math.floor((np.size(sub2,0)-totalbii[x-y,0])/2)
                dif2 = math.floor((np.size(sub2,1)-totalbii[x-y,1])/2)
                sub2 = sub2[dif1:np.size(sub2,0)-dif1, dif2:np.size(sub2,1)-dif2]
            continue
            sub2 =(signal.convolve2d(sub2,Gaussian))
            sub2 = sub2[1:np.size(sub2,0)-1,1:np.size(sub2,1)-1]
            sub2 = normal(sub2)
            bi1 = sub2
            #sub2=np.float64(sub2)
        sub3[:,(x+1)*128:(x+2)*128] = sub2
            
    return sub3

def preprocessing(Inputimage):
    """
    def edge_detect(Inputimage,mask1,mask2):
        sobel1 = signal.convolve2d(Inputimage,mask1)
        sobel1 = sobel1[1:np.size(sobel1,0)-1,1:np.size(sobel1,1)-1]
        sobel2 = signal.convolve2d(Inputimage,mask2)
        sobel2 = sobel2[1:np.size(sobel2,0)-1,1:np.size(sobel2,1)-1]
        s = np.sqrt(sobel1**2 + sobel2**2)
        s = s*255/np.max(s)
        return s
    
    def normal(Inputimage):
        min_1 = np.min(Inputimage, axis = None)
        m = (Inputimage-min_1)*255./(np.max(Inputimage, axis = None)-min_1)
        return m
    
    def pyramid(Inputimage):
            
        Gaussian = np.array([[2,13,2],[13,40,13],[2,13,2]])
        totalbii = np.zeros([8,2])
        bi = np.uint8(Inputimage)
        width = bi.shape[0]
        height = bi.shape[1]
        sub3 = np.zeros([128, 128*8])
        for x in range (0,7,1):
            width = width//2
            height = height//2
            sub1 = imresize(bi, (width,height))
            sub1 = signal.convolve2d(sub1, Gaussian)
            sub1 = sub1[1:np.size(sub1,0)-1,1:np.size(sub1,1)-1]
            sub1 = normal(sub1)
            bi = sub1
            bi1 = sub1
            totalbi = np.array([[bi.shape[0],bi.shape[1]]])
            totalbii[0:1,:]=np.array([[Inputimage.shape[0],Inputimage.shape[1]]])
            totalbii[x+1:x+2,:] = totalbi
            width1 = bi1.shape[0]
            height1 = bi1.shape[1] 
            for y in range(0,x+1,1):
                
                width1=width1*2
                height1=height1*2
                sub2 = imresize(bi1,(width1, height1),interp='bilinear')
                Q = (np.size(sub2,axis=0) != totalbii[x-y][0] or np.size(sub2,axis=1) != 1)
                W = (np.mod(np.size(sub2,axis=0)-totalbii[x-y][0], 2) == 0)
                E = (np.mod(np.size(sub2,axis=1)-totalbii[x-y][0], 2) == 0)
                if Q and W and E:
                    #if ((sub2.shape[0] != totalbi[x+1-y,0] or sub2.shape[1] != 1) and (np.mod(sub2.shape[0]-totalbi[x+1-y,0],1)==0) and (np.mod(sub2.shape[1]-totalbi[x+1-y,0],1)==0):
                    dif1 = math.floor((np.size(sub2,0)-totalbii[x-y,0])/2)
                    dif2 = math.floor((np.size(sub2,1)-totalbii[x-y,1])/2)
                    sub2 = sub2[dif1:np.size(sub2,0)-dif1, dif2:np.size(sub2,1)-dif2]
                    continue
                sub2 =(signal.convolve2d(sub2,Gaussian))
                sub2 = sub2[1:np.size(sub2,0)-1,1:np.size(sub2,1)-1]
                sub2 = normal(sub2)
                bi1 = sub2
                #sub2=np.float64(sub2)
                sub3[:,0:128] = Inputimage
                sub3[:,(x+1)*128:(x+2)*128] = sub2
                return sub3
        """
    img1 = np.double(Inputimage)
    r = img1[:,:,0]
    g = img1[:,:,1]
    b = img1[:,:,2]
    
    realR =r - (b+g)/2
    realR[realR<0]=0
    rr = normal(realR)
    
    realG =g - (b+r)/2
    realG[realG<0]=0
    gg = normal(realG)
    
    realB =b - (r+g)/2
    realB[realB<0]=0
    bb = normal(realB)
    
    Y = ((r+g)/2) - b
    Y[Y<0]=0
    yy = normal(Y)
    
    totalcolor = r + g + b
    #R = r/totalcolor
    #G = g/totalcolor
    #B = b/totalcolor
    i = (totalcolor)/3
    
    mask1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    mask2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Sobel1 = edge_detect(r,mask1,mask2)
    Sobel2 = edge_detect(g,mask1,mask2)
    Sobel3 = edge_detect(b,mask1,mask2)
    Sobel2[Sobel1>Sobel2] = Sobel1[Sobel1>Sobel2]
    Sobel3[Sobel2>Sobel3] = Sobel2[Sobel2>Sobel3]
    
    #가우시안 피라미드 (함수 참고)
    itotal = pyramid(i)
    etotal = pyramid(Sobel3)
    rtotal = pyramid(rr)
    gtotal = pyramid(gg)
    btotal = pyramid(bb)
    ytotal = pyramid(yy)  
    imsave('itotal.jpg',itotal)
    imsave('etotal.jpg',etotal)
    imsave('rtotal.jpg',rtotal)
    imsave('gtotal.jpg',gtotal)
    imsave('btotal.jpg',btotal)
    imsave('ytotal.jpg',ytotal)
    
    
    I1 = np.zeros([128,128])
    I2 = np.zeros([128,128])
    I3 = np.zeros([128,128])
    I1 = (abs(itotal[:,128:128*2] - itotal[:,128*3:128*4]) +abs(itotal[:,128:128*2] - itotal[:,128*4:128*5]))
    I1 = normal(I1)
    
    I2 = (abs(itotal[:,128*2+1:128*3+1] - itotal[:,128*4+1:128*5+1]) +abs(itotal[:,128*2+1:128*3+1] - itotal[:,128*5+1:128*6+1]))
    I2 = normal(I2)
    
    I3 = (abs(itotal[:,128*3+1:128*4+1] - itotal[:,128*5+1:128*6+1]) +abs(itotal[:,128*3+1:128*4+1] - itotal[:,128*6+1:128*7+1]))
    I3 = normal(I3)
    
    I = (I1 +I2 +I3)
    I = normal(I)

    
    E1 = (abs(etotal[:,128:128*2] - etotal[:,128*3:128*4]) +abs(etotal[:,128:128*2] - etotal[:,128*4:128*5]))
    E1 = normal(E1)
    
    E2 = (abs(etotal[:,128*2+1:128*3+1] - etotal[:,128*4+1:128*5+1]) +abs(etotal[:,128*2+1:128*3+1] - etotal[:,128*5+1:128*6+1]))
    E2 = normal(E2)
    
    E3 = (abs(etotal[:,128*3+1:128*4+1] - etotal[:,128*5+1:128*6+1]) +abs(etotal[:,128*3+1:128*4+1] - etotal[:,128*6+1:128*7+1]))
    E3 = normal(E3)
    
    E = (E1 +E2 +E3)
    E = normal(E)
    
    RG11 = abs((rtotal[:,128:128*2] - gtotal[:,128:128*2]) -(gtotal[:,128*2+1:128*3+1] - rtotal[:,128*2+1:128*3+1]))
    RG11 = normal(RG11);
    RG12 = abs((rtotal[:,128:128*2] - gtotal[:,128:128*2]) -(gtotal[:,128*3+1:128*4+1] - rtotal[:,128*3+1:128*4+1]))
    RG12 = normal(RG12)
    RG1 = RG11 +RG12
    RG1 = normal(RG1)
    
    BY11 = abs((btotal[:,128:128*2] - ytotal[:,128:128*2]) -(ytotal[:,128*2+1:128*3+1] - btotal[:,128*2+1:128*3+1]))
    BY11 = normal(BY11);
    BY12 = abs((btotal[:,128:128*2] - ytotal[:,128:128*2]) -(ytotal[:,128*3+1:128*4+1] - btotal[:,128*3+1:128*4+1]))
    BY12 = normal(BY12)
    BY1 = BY11 +BY12
    BY1 = normal(BY1)
    
    C1 = RG1 + BY1
    C1 = normal(C1)
    
    RG21 = abs((rtotal[:,128*2+1:128*3+1] - gtotal[:,128*2+1:128*3+1]) -(gtotal[:,128*4+1:128*5+1] - rtotal[:,128*4+1:128*5+1]))
    RG21 = normal(RG21)
    RG22 = abs((rtotal[:,128*2+1:128*3+1] - gtotal[:,128*2+1:128*3+1]) -(gtotal[:,128*5+1:128*6+1] - rtotal[:,128*5+1:128*6+1]))
    RG22 = normal(RG22)
    RG2 = RG21 +RG22
    RG2 = normal(RG2)
    
    BY21 = abs((btotal[:,128*2+1:128*3+1] - ytotal[:,128*2+1:128*3+1]) -(ytotal[:,128*4+1:128*5+1] - btotal[:,128*4+1:128*5+1]))
    BY21 = normal(BY21)
    BY22 = abs((btotal[:,128*2+1:128*3+1] - ytotal[:,128*2+1:128*3+1]) -(ytotal[:,128*5+1:128*6+1] - btotal[:,128*5+1:128*6+1]))
    BY22 = normal(BY22)
    BY2 = BY21 +BY22
    BY2 = normal(BY2)
    
    C2 = RG2 + BY2
    C2 = normal(C2)
    
    RG31 = abs((rtotal[:,128*3+1:128*4+1] - gtotal[:,128*3+1:128*4+1]) -(gtotal[:,128*5+1:128*6+1] - rtotal[:,128*5+1:128*6+1]))
    RG31 = normal(RG31)
    RG32 = abs((rtotal[:,128*3+1:128*4+1] - gtotal[:,128*3+1:128*4+1]) -(gtotal[:,128*6+1:128*7+1] - rtotal[:,128*6+1:128*7+1]))
    RG32 = normal(RG32)
    RG3 = RG31 +RG32
    RG3 = normal(RG3)
    
    BY31 = abs((btotal[:,128*3+1:128*4+1] - ytotal[:,128*3+1:128*4+1]) -(ytotal[:,128*5+1:128*6+1] - btotal[:,128*5+1:128*6+1]))
    BY31 = normal(BY31)
    BY32 = abs((btotal[:,128*3+1:128*4+1] - ytotal[:,128*3+1:128*4+1]) -(ytotal[:,128*6+1:128*7+1] - btotal[:,128*6+1:128*7+1]))
    BY32 = normal(BY32)
    BY3 = BY31 +BY32
    BY3 = normal(BY3)
    
    C3 = RG3 + BY3
    C3 = normal(C3)
    
    C = (C1+C2+C3)
    C = normal(C)
    
    SM = 0.3*I + 0.4*E + 0.3*C
    SM = normal(SM)
    imsave('Itensity_feature_map.jpg', I)
    imsave('Edge_feature_map.jpg', E)
    imsave('Color_feature_map.jpg', C)
    imsave('Saliency_feature_map.jpg', SM)
        
    img33 = im.filter(ImageFilter.GaussianBlur(10))
    img3 = np.uint8(img33)
    #ha = np.ones((10,10), dtype = float) / 100
    #img3 = scipy.ndimage.convolve(img1,ha, mode ='symmetric')
    
    #ha1 = np.ones((5,5), dtype = float) / 25
    #img2 = scipy.ndimage.convolve(input0,ha1, mode ='symmetric')
    img22 = im.filter(ImageFilter.GaussianBlur(5))
    img2 = np.uint8(img22)
    
    tmp_SM1 = np.zeros((128,128))
    tmp_SM2 = np.zeros((128,128))
    
    SIZE1 = np.array([128,128])
    for m in range(0,SIZE1[0]):
        for n in range(0,SIZE1[1]):
            if SM[m,n] <= THRESHOULD_SM2:
                tmp_SM1[m,n] = 0
            else:
                tmp_SM1[m,n] = 1
            if SM[m,n] <= THRESHOULD_SM1:
                tmp_SM2[m,n] = 0
            else:
                tmp_SM2[m,n] = 1
                
                
                
    tmp_SM3 = tmp_SM2 - tmp_SM1
    tmp_SM4 = tmp_SM2 + tmp_SM3
    
    out_img = np.ones((128, 128, 3))
    for o in range(0,3):
        for m in range(0,SIZE1[0]):
            for n in range(0,SIZE1[1]):
                if tmp_SM4[m,n] == 1:
                    out_img[m,n,o] = img1[m,n,o]
                elif tmp_SM4[m,n] == 2:
                    out_img[m,n,o] = img3[m,n,o]
                else:
                    out_img[m,n,o] = img2[m,n,o]

    out_images = np.uint8(out_img)
    imsave('tmp_SM3.jpg', tmp_SM3)
    imsave('tmp_SM4.jpg', tmp_SM4)
    imsave('img2.jpg', img2)
    imsave('img3.jpg', img3)
    imsave('test/Preprocessing_Image.jpg', out_images)
    return out_images

def CNN():
    img_width, img_height = 128, 128
    validation_data_dir = 'test'
    nb_validation_samples = 1
    batch_size=1
    if K.image_data_format() == 'channels_first':    
        input_shape = (3, img_width, img_height) # 3->1
    else:
        input_shape = (img_width, img_height, 3) # 3->1
        
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape,
                             strides=(1, 1), padding='same', init='glorot_uniform', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.02))
    
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.015))
    
    
    # 펴기 3D피쳐맵 -> 1D
    model.add(Flatten())
    # MLP
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.load_weights('Superclass_weights.h5')
    
    test_datagen = ImageDataGenerator(
            rescale=1. / 255)
    
    
    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            color_mode = 'rgb',
            batch_size=batch_size,
            class_mode='categorical')
    #        shuffle=False) # past_use : rgb or graysclale, binary or categorical
    
    
    print("Generating test predictions...")
    preds = model.predict_generator(validation_generator, steps=nb_validation_samples//batch_size)
    
    
    for i in range(0,5):
        if preds[0, i] >= np.max(preds): 
            print("Image_class =",validation_generator.classes, "Class =", i ,", Acc = {:.2%} --> correct  ".format(preds[0][i])) 
        else: 
            print("Image_class =",validation_generator.classes,"Class =", i ,", Acc = {:.2%} --> Not correct ".format(preds[0][i]))
        if validation_generator.classes == i and preds[0][i] < 0.5:
            print("                About image...max accuracy = {:.2%}".format(np.max(preds)),"-->> -The recognition is wrong-")    
            print("                The image is wrong or not recognized")
            print("                Filename =", validation_generator.filenames)
        if validation_generator.classes == i and preds[0][i] >= 0.5:
            print("                The recognition is well done")
            print("                Filename =", validation_generator.filenames)
    return preds
def image_CNN():
    out = CNN
    canvas.create_text(100,300,text=out)
    window.update()
  
                        
def image_processing():
    global im, tk_img1
    out = preprocessing(im)
    tk_img1 = ImageTk.PhotoImage(Image.fromarray(out))
    canvas.create_image(300, 100, image=tk_img1)    
    window.update()
            
            # 영사처리 메뉴에서 “열기”를 선택하였을 때 호출되는 함수
"""
def image_CNN():
    global im, tk_img2
    out = im.filter(ImageFilter.BLUR)
    tk_img2 = ImageTk.PhotoImage(out)
    canvas.create_image(500, 100, image=tk_img2)
    window.update()
"""
# 윈도우를 생성한다. 
window = tk.Tk()
canvas = tk.Canvas(window, width=1000, height=1000)
canvas.pack()

# 메뉴를 생성한다. 
menubar = tk.Menu(window)
filemenu = tk.Menu(menubar)
ipmenu = tk.Menu(menubar)
filemenu.add_command(label="열기", command=open)
filemenu.add_command(label="종료", command=quit)
ipmenu.add_command(label="Preprocessing", command=image_processing)
ipmenu.add_command(label="CNN", command=image_CNN)
menubar.add_cascade(label="파일", menu=filemenu)
menubar.add_cascade(label="영상처리", menu=ipmenu)

window.config(menu=menubar)
window.mainloop()

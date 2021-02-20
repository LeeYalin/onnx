#!/usr/bin/env python
# coding: utf-8
import onnxruntime as xr

import sys
#reload(sys)
#sys.setdefaultxxxx("utf8")

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import cv2
import numpy as np
from easydict import EasyDict as edict
from yaml import load

onnxruntime = True
opencv_use = False

model_name = 'v3.onnx'
########  onnxruntime 调用  ###########
if onnxruntime:
    sess = xr.InferenceSession(model_name)#加载模型 model_best_two_auxiliary_losses
    
    input_name0 = sess.get_inputs()[0].name#获取输入层的名字，如果有多个输入，需要按照顺序都获取到
    model = sess.get_modelmeta()
    
########  opencv调用  #########
if opencv_use:
    net = cv2.dnn.readNetFromONNX(model_name)







videofile = 'VID_20210112_174111.mp4'

videoCapture = cv2.VideoCapture(videofile)
#size = ((int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))+20)*3, int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#videoWriter = cv2.VideoWriter('kk_result.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

succ, img = videoCapture.read()
cnt = 1
while succ:
    h, w, _ = img.shape
    #cv2.imwrite("7777777777.png", img)
    img = cv2.copyMakeBorder(img,0,0,82,82,cv2.BORDER_CONSTANT,value=[0,0,0])
    img = cv2.resize(img, (288, 384))
    img_f = img - (103.94, 116.78, 123.68)
    
    img_f = img_f*0.017
    
    show_image = np.uint8(img/np.max(img)*255)
    cv2.imshow('1', show_image)
    #cv2.waitKey(0)
    # show_image = cv2.cvtColor(show_image, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('img',np.uint8(img/np.max(img)*255))
   
    img_f = np.array(img_f)
    img_f = np.array(np.float32(img_f))
    img_f = np.expand_dims(img_f, 0)
   
    #output_names为输出层名字，需要事先被确认好
    
    img_f = img_f.transpose(0, 3, 1, 2) # kang
    
    ########  opencv infer  #########
    if opencv_use:
        start = cv2.getTickCount()
        #blob = cv2.dnn.blobFromImage(img_f, size=(388, 284), crop=False)
        # Run a model
        net.setInput(img_f)
        out = net.forward()
        out = list(out)
        mask_cv = np.argmax(out, 1).squeeze().astype(np.int8)
        end = cv2.getTickCount()
        time = (end - start) / cv2.getTickFrequency()
        print("opencv time is: " + str(time) + "s")
        cv2.imshow('mask_cv', np.uint8(mask_cv*255))
    ########  onnxruntime infer  ###########
    if onnxruntime:
        start = cv2.getTickCount()
        # res = sess.run(output_names = ["Concat__87"],input_feed = {input_name0:img})
        res = sess.run(output_names = ["output1"],input_feed = {input_name0:img_f})
        # mask = np.argmax(ort_outs[0], 1).squeeze().astype(np.int8)
        mask = np.argmax(res[0], 1).squeeze().astype(np.int8)
        #cv2.imwrite("result.jpg",mask*255)
        end = cv2.getTickCount()
        time = (end - start) / cv2.getTickFrequency()
        print("onnxruntime time is: " + str(time) + "s")
        cv2.imshow('mask_onnx', np.uint8(mask*255))
    
    cv2.waitKey(1)
    succ, img = videoCapture.read()
    cnt += 1
#end = cv2.getTickCount()
#time = (end - start) / cv2.getTickFrequency()
#fps = cnt/time
#print("fps is: " + str(fps) + "s")

import cv2 as cv
import numpy as np
import time
from cv2 import COLOR_BGR2HSV
from itertools import product
import cvzone

# confidence and nms threshold to be applied on model
Conf_threshold = 0.4
NMS_threshold = 0.4

# colours to be used for labelling classes
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

#HSV colour ranges of maximum/minimum speed limit signs
lower_red = np.array([160,160,20])
upper_red = np.array([179,255,255])
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

#png instructions to be overlaid
Min20Message = cv.imread("min20.png", cv.IMREAD_UNCHANGED)
Min20Message = cv.resize(Min20Message, (0,0), None, 0.3,0.3)
Min30Message = cv.imread("min30.png", cv.IMREAD_UNCHANGED)
Min30Message = cv.resize(Min30Message, (0,0), None, 0.3,0.3)
Min40Message = cv.imread("min40.png", cv.IMREAD_UNCHANGED)
Min40Message = cv.resize(Min40Message, (0,0), None, 0.3,0.3)
Min50Message = cv.imread("min50.png", cv.IMREAD_UNCHANGED)
Min50Message = cv.resize(Min50Message, (0,0), None, 0.3,0.3)
Min60Message = cv.imread("min60.png", cv.IMREAD_UNCHANGED)
Min60Message = cv.resize(Min60Message, (0,0), None, 0.3,0.3)
Min70Message = cv.imread("min70.png", cv.IMREAD_UNCHANGED)
Min70Message = cv.resize(Min70Message, (0,0), None, 0.3,0.3)
Min80Message = cv.imread("min80.png", cv.IMREAD_UNCHANGED)
Min80Message = cv.resize(Min80Message, (0,0), None, 0.3,0.3)
MinSpeedHUD = cv.imread("MinSpeedHUD.png", cv.IMREAD_UNCHANGED)
MinSpeedHUD = cv.resize(MinSpeedHUD, (0,0), None, 0.3,0.3)
Max20Message = cv.imread("max20.png", cv.IMREAD_UNCHANGED)
Max20Message = cv.resize(Max20Message, (0,0), None, 0.3,0.3)
Max30Message = cv.imread("max30.png", cv.IMREAD_UNCHANGED)
Max30Message = cv.resize(Max30Message, (0,0), None, 0.3,0.3)
Max40Message = cv.imread("max40.png", cv.IMREAD_UNCHANGED)
Max40Message = cv.resize(Max40Message, (0,0), None, 0.3,0.3)
Max50Message = cv.imread("max50.png", cv.IMREAD_UNCHANGED)
Max50Message = cv.resize(Max50Message, (0,0), None, 0.3,0.3)
Max60Message = cv.imread("max60.png", cv.IMREAD_UNCHANGED)
Max60Message = cv.resize(Max60Message, (0,0), None, 0.3,0.3)
Max70Message = cv.imread("max70.png", cv.IMREAD_UNCHANGED)
Max70Message = cv.resize(Max70Message, (0,0), None, 0.3,0.3)
Max80Message = cv.imread("max80.png", cv.IMREAD_UNCHANGED)
Max80Message = cv.resize(Max80Message, (0,0), None, 0.3,0.3)
MaxSpeedHUD = cv.imread("MaxSpeedHUD.png", cv.IMREAD_UNCHANGED)
MaxSpeedHUD = cv.resize(MaxSpeedHUD, (0,0), None, 0.3,0.3)

#model configuration
width = 416
height = 416
weights = "backup/yolov4-tiny-obj_final.weights"
cfg = "cfg/yolov4-tiny-obj.cfg"
class_name = []
with open('data/obj.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
print(class_name)
net = cv.dnn.readNet(weights, cfg)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

#set input parameters of model
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
HSVImage = []

# define speed sign labels
speedsigns = ['speed sign 20mph', 'speed sign 30mph', 'speed sign 40mph',
             'speed sign 50mph', 'speed sign 60mph', 'speed sign 70mph', 'speed sign 80mph']

#read in image
cap = cv.imread('data/ss12.jpg')

#overlay canvas for HUD
cap = cv.resize(cap,(1080,720),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
cap = cvzone.overlayPNG(cap, MinSpeedHUD, [0,667])
cap = cvzone.overlayPNG(cap, MaxSpeedHUD, [792,667])
classes, scores, boxes = model.detect(cap, Conf_threshold, NMS_threshold)
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = "%s : %.2f" % (class_name[classid[0]], score)

#look for speed signs specifically with a confidence higher than 60%
    if class_name[classid[0]] in speedsigns and score > 0.6:
       # box_width = box[2]
        #box_height = box[3]
        #print(box_width)
        #print(box_height)
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]

        # get rectangle surrounding speed sign and transform to 416x416 window
        srcSpeedSign  = np.float32([[box[0],box[1]],[box[2],box[1]],[box[0],box[3]],[box[2],box[3]]])
        dstSpeedSign = np.float32([[0,0], [width,0], [0,height], [width,height]])
        SpeedSignPT = cv.getPerspectiveTransform(srcSpeedSign, dstSpeedSign)
        warpedSpeedSign = cv.warpPerspective(cap, SpeedSignPT, (width,height))
        #count number of coloured pixels
        kernel = np.ones((5, 5), "uint8")
        #Max Speed (Red)
        HSVImage = cv.cvtColor(warpedSpeedSign, cv.COLOR_BGR2HSV)
        MaxSpeedMask = cv.inRange(HSVImage, lower_red, upper_red)
        MaxSpeedMask = cv.dilate(MaxSpeedMask, kernel)
        RedPixels = cv.countNonZero(MaxSpeedMask)
        print("Red Pixels = ", RedPixels)
        #Min Speed (Blue)
        MinSpeedMask = cv.inRange(HSVImage, lower_blue ,upper_blue)
        MinSpeedMask = cv.dilate(MinSpeedMask, kernel)
        BluePixels = cv.countNonZero(MinSpeedMask)
        print("Blue Pixels = ", BluePixels)
        
        # adjust labels based on state of label 
        if class_name[classid[0]] == "speed sign 20mph":
            if RedPixels > BluePixels:
                SpeedSignState = "(Max Speed)"
                MaxSpeedHUD = Max20Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            else:
                SpeedSignState = "(Min Speed)"
                MinSpeedHUD = Min20Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        elif class_name[classid[0]] == "speed sign 30mph":
            if RedPixels > BluePixels:
                SpeedSignState = "(Max Speed)"
                MaxSpeedHUD = Max30Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            else:
                SpeedSignState = "(Min Speed)"
                MinSpeedHUD = Min30Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        elif class_name[classid[0]] == "speed sign 40mph":
            if RedPixels > BluePixels:
                SpeedSignState = "(Max Speed)"
                MaxSpeedHUD = Max40Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            else:
                SpeedSignState = "(Min Speed)"
                MinSpeedHUD = Min40Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        elif class_name[classid[0]] == "speed sign 50mph":
            if RedPixels > BluePixels:
                SpeedSignState = "(Max Speed)"
                MaxSpeedHUD = Max50Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            else:
                SpeedSignState = "(Min Speed)"
                MinSpeedHUD = Min50Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        elif class_name[classid[0]] == "speed sign 60mph":
            if RedPixels > BluePixels:
                SpeedSignState = "(Max Speed)"
                MaxSpeedHUD = Max60Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            else:
                SpeedSignState = "(Min Speed)"
                MinSpeedHUD = Min60Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        elif class_name[classid[0]] == "speed sign 70mph":
            if RedPixels > BluePixels:
                SpeedSignState = "(Max Speed)"
                MaxSpeedHUD = Max70Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            else:
                SpeedSignState = "(Min Speed)"
                MinSpeedHUD = Min70Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        elif class_name[classid[0]] == "speed sign 80mph":
            if RedPixels > BluePixels:
                SpeedSignState = "(Max Speed)"
                MaxSpeedHUD = Max80Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            else:
                SpeedSignState = "(Min Speed)"
                MinSpeedHUD = Min80Message
                cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(cap, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        cap = cvzone.overlayPNG(cap, MinSpeedHUD, [0,667])
        cap = cvzone.overlayPNG(cap, MaxSpeedHUD, [792,667])
cv.imshow("Test", cap)
cv.imshow("warpedTransform", warpedSpeedSign)
cv.waitKey(0)
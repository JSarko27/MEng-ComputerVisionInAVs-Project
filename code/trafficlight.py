import cv2 as cv
from cv2 import perspectiveTransform
import numpy as np
import cvzone
#from __future__ import print_function


# define NMS and confidence threshold for model
Conf_threshold = 0.4
NMS_threshold = 0.4
# specify colours for labelling objects
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

#colour ranges of traffic lights
lower_red = np.array([160,160,20])
upper_red = np.array([179,255,255])
lower_amber = np.array([20,200,230])
upper_amber = np.array([30,255,255])
lower_green = np.array([36,140,140])
upper_green = np.array([86,255,255])


RedLightCtr = 0
AmberLightCtr = 0
GreenLightCtr = 0

#HUD
HUDBackground = cv.imread("HUDBackground.png", cv.IMREAD_UNCHANGED)
HUDBackground = cv.resize(HUDBackground, (0,0), None, 0.3,0.3)
RedLightMessage = cv.imread("RedLightMessage.png", cv.IMREAD_UNCHANGED)
RedLightMessage = cv.resize(RedLightMessage, (0,0), None, 0.3,0.3)
GreenLightMessage = cv.imread("GreenLightMessage.png", cv.IMREAD_UNCHANGED)
GreenLightMessage = cv.resize(GreenLightMessage, (0,0), None, 0.3,0.3)
MoreRedThanGreenMessage = cv.imread("MoreRedThanGreen.png", cv.IMREAD_UNCHANGED)
MoreRedThanGreenMessage = cv.resize(MoreRedThanGreenMessage, (0,0), None, 0.3,0.3)
MoreGreenThanRedMessage = cv.imread("MoreGreenThanRed.png", cv.IMREAD_UNCHANGED)
MoreGreenThanRedMessage = cv.resize(MoreGreenThanRedMessage, (0,0), None, 0.3,0.3)
width = 1080
height = 720

#model configuration
weights = "backup/yolov4-tiny-obj_final.weights"
cfg = "cfg/yolov4-tiny-obj.cfg"
class_name = []
with open('data/obj.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet(weights, cfg)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

#model input parameters
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
HSVImage = []
# traffic light states
TrafficLightColour = ['(Red)', '(Amber)', '(Green)']

#read input image
cap = cv.imread('data/pic12.jpg')
cap = cv.resize(cap,(1080,720),fx=0,fy=0, interpolation = cv.INTER_CUBIC)

classes, scores, boxes = model.detect(cap, Conf_threshold, NMS_threshold)
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = "%s : %.2f" % (class_name[classid[0]], score)
    
    # look for traffic lights with more than 60% confidence
    if class_name[classid[0]] == "traffic light" and score > 0.6:

        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        # take coordinates of traffic light and warp with warp perspective
        srcTrafficLightFloat  = np.float32([[box[0],box[1]],[box[2],box[1]],[box[0],box[3]],[box[2],box[3]]])
        dstTrafficLightFloat = np.float32([[0,0], [width,0], [0,height], [width,height]])
        TrafficLightPT = cv.getPerspectiveTransform(srcTrafficLightFloat, dstTrafficLightFloat)
        warpedTrafficLight = cv.warpPerspective(cap, TrafficLightPT, (width,height))
        
        #count number of coloured pixels
        kernel = np.ones((5, 5), "uint8")
        #red
        HSVImage = cv.cvtColor(warpedTrafficLight, cv.COLOR_BGR2HSV)
        #red
        RedLightMask = cv.inRange(HSVImage, lower_red, upper_red)
        RedLightMask = cv.dilate(RedLightMask, kernel)
        RedLightPixels = cv.countNonZero(RedLightMask)
        print("Red Light Pixels = ", RedLightPixels)
        #amber
        AmberLightMask = cv.inRange(HSVImage, lower_amber,upper_amber)
        AmberLightMask = cv.dilate(AmberLightMask, kernel)
        AmberLightPixels = cv.countNonZero(AmberLightMask)
        print("Amber Light Pixels = ",AmberLightPixels)
        #green
        GreenLightMask = cv.inRange(HSVImage, lower_green ,upper_green)
        GreenLightMask = cv.dilate(GreenLightMask, kernel)
        GreenLightPixels = cv.countNonZero(GreenLightMask)
        print("Green Light Pixels = ", GreenLightPixels)

        #define red colour
        if RedLightPixels > GreenLightPixels and RedLightPixels > AmberLightPixels:
            TrafficLightState = TrafficLightColour[0]
            RedLightCtr +=1
            cap = cvzone.overlayPNG(cap, RedLightMessage, [325,0])
            cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
            cv.putText(cap, 'Traffic Light: ' + TrafficLightState + ' %.2f' % score, (box[0], box[1]-10),
                cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
        #define actions for amber   
        elif AmberLightPixels > RedLightPixels and AmberLightPixels > GreenLightPixels:
            TrafficLightState = TrafficLightColour[1]
            AmberLightCtr +=1
            cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
            cv.putText(cap, 'Traffic Light: ' + TrafficLightState + ' %.2f' % score, (box[0], box[1]-10),
                cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
        #define green colour
        elif GreenLightPixels > RedLightPixels and GreenLightPixels > AmberLightPixels: 
            TrafficLightState = TrafficLightColour[2]
            GreenLightCtr +=1
            cap = cvzone.overlayPNG(cap, GreenLightMessage, [325,0])
            cv.rectangle(cap, (box[0],box[1]), (box[2],box[3]), color, 1)
            cv.putText(cap, 'Traffic Light: ' + TrafficLightState + ' %.2f' % score, (box[0], box[1]-10),
                cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)       
cv.imshow("Test", cap)
cv.waitKey(0)
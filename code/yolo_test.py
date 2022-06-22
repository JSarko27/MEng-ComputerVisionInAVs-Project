import cv2 as cv
from cv2 import COLOR_BGR2HSV
import numpy as np
import time
import math
import cvzone

# set confidence and NMS thresholds for model
Conf_threshold = 0.45
NMS_threshold = 0.4
#set colours to be used for labelling
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

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

#HSV colour ranges
lower_red = np.array([135,40,20])
upper_red = np.array([179,255,255])
lower_amber = np.array([12,176,49])
upper_amber = np.array([31,255,255])
lower_green = np.array([33,119,60])
upper_green = np.array([90,255,255])
lower_blue = np.array([110,120,50])
upper_blue = np.array([130,255,255])

width = 200
height = 600

#model configuration
weights = "backup/yolov4-tiny-obj_final.weights"
cfg = "cfg/yolov4-tiny-obj.cfg"

class_name = []
speedsigns = ['speed sign 20mph', 'speed sign 30mph', 'speed sign 40mph',
             'speed sign 50mph', 'speed sign 60mph', 'speed sign 70mph', 'speed sign 80mph']
TrafficLightColour = ['(Red)', '(Amber)', '(Green)']
with open('data/obj.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet(weights, cfg)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

#set model input parameters
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#read input video
cap = cv.VideoCapture('data/vid11.mp4')
starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()
    try:
        b = cv.resize(frame,(1080,720),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
    except cv.error as e:
        print('Invalid frame!')
        cv.waitKey()
    
    #overlay HUD on video
    b = cvzone.overlayPNG(b, HUDBackground, [325,0])
    b = cvzone.overlayPNG(b, MinSpeedHUD, [0,667])
    b = cvzone.overlayPNG(b, MaxSpeedHUD, [792,667])
    frame_counter += 1
    if ret == False:
        break
    classes, scores, boxes = model.detect(b, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_name[classid[0]], score)
        
        # look for traffic lights with 60%+ confidence
        if class_name[classid[0]] == "traffic light" and score > 0.6:
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            
            #warp traffic light
            srcTrafficLightFloat  = np.float32([[box[0],box[1]],[box[2],box[1]],[box[0],box[3]],[box[2],box[3]]])
            dstTrafficLightFloat = np.float32([[0,0], [width,0], [0,height], [width,height]])
            TrafficLightPT = cv.getPerspectiveTransform(srcTrafficLightFloat, dstTrafficLightFloat)
            warpedTrafficLight = cv.warpPerspective(b, TrafficLightPT, (width,height))
            #count number of coloured pixels
            kernel = np.ones((5, 5), "uint8")
            #red
            HSVImage = cv.cvtColor(warpedTrafficLight, cv.COLOR_BGR2HSV)
            lower_red = np.array([160,160,20])
            RedLightMask = cv.inRange(HSVImage, lower_red, upper_red)
            RedLightMask = cv.dilate(RedLightMask, kernel)
            RedLightPixels = cv.countNonZero(RedLightMask)
            #print("Red Light Pixels = ", RedLightPixels)
            #amber
            AmberLightMask = cv.inRange(HSVImage, lower_amber,upper_amber)
            AmberLightMask = cv.dilate(AmberLightMask, kernel)
            AmberLightPixels = cv.countNonZero(AmberLightMask)
            #print("Amber Light Pixels = ",AmberLightPixels)
            #green
            GreenLightMask = cv.inRange(HSVImage, lower_green ,upper_green)
            GreenLightMask = cv.dilate(GreenLightMask, kernel)
            GreenLightPixels = cv.countNonZero(GreenLightMask)
            #print("Green Light Pixels = ", GreenLightPixels)

            #define when traffic light is red
            if RedLightPixels > GreenLightPixels and RedLightPixels > AmberLightPixels:
                TrafficLightState = TrafficLightColour[0]
                b = cvzone.overlayPNG(b, RedLightMessage, [325,0])
                cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(b, 'Traffic Light: ' + TrafficLightState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            #define when traffic light is amber
            elif AmberLightPixels > RedLightPixels and AmberLightPixels > GreenLightPixels:
                TrafficLightState = TrafficLightColour[1]
                cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(b, 'Traffic Light: ' + TrafficLightState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            #define when traffic light is green
            elif GreenLightPixels > RedLightPixels and GreenLightPixels > AmberLightPixels: 
                TrafficLightState = TrafficLightColour[2]
                b = cvzone.overlayPNG(b, GreenLightMessage, [325,0])
                cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                cv.putText(b, 'Traffic Light: ' + TrafficLightState + ' %.2f' % score, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # look for speed signs with confidence greater tham 60%
        elif class_name[classid[0]] in speedsigns and score > 0.6:
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            
            #warp speed sign
            srcSpeedSign  = np.float32([[box[0],box[1]],[box[2],box[1]],[box[0],box[3]],[box[2],box[3]]])
            dstSpeedSign = np.float32([[0,0], [width,0], [0,height], [width,height]])
            SpeedSignPT = cv.getPerspectiveTransform(srcSpeedSign, dstSpeedSign)
            warpedSpeedSign = cv.warpPerspective(b, SpeedSignPT, (width,height))
            #count number of coloured pixels
            kernel = np.ones((5, 5), "uint8")
            #Max Speed (Red)
            lower_red = np.array([135,40,20])
            HSVImage = cv.cvtColor(warpedSpeedSign, cv.COLOR_BGR2HSV)
            MaxSpeedMask = cv.inRange(HSVImage, lower_red, upper_red)
            MaxSpeedMask = cv.dilate(MaxSpeedMask, kernel)
            RedPixels = cv.countNonZero(MaxSpeedMask)
            #print("Red Pixels = ", RedPixels)
            #Min Speed (Blue)
            MinSpeedMask = cv.inRange(HSVImage, lower_blue ,upper_blue)
            MinSpeedMask = cv.dilate(MinSpeedMask, kernel)
            BluePixels = cv.countNonZero(MinSpeedMask)
            #print("Blue Pixels = ", BluePixels)
            #define state of speed sign 
            if class_name[classid[0]] == "speed sign 20mph":
                if RedPixels > BluePixels:
                    SpeedSignState = "(Max Speed)"
                    MaxSpeedHUD = Max20Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    SpeedSignState = "(Min Speed)"
                    MinSpeedHUD = Min20Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            elif class_name[classid[0]] == "speed sign 30mph":
                if RedPixels > BluePixels:
                    SpeedSignState = "(Max Speed)"
                    MaxSpeedHUD = Max30Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    SpeedSignState = "(Min Speed)"
                    MinSpeedHUD = Min30Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            elif class_name[classid[0]] == "speed sign 40mph":
                if RedPixels > BluePixels:
                    SpeedSignState = "(Max Speed)"
                    MaxSpeedHUD = Max40Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    SpeedSignState = "(Min Speed)"
                    MinSpeedHUD = Min40Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            elif class_name[classid[0]] == "speed sign 50mph":
                if RedPixels > BluePixels:
                    SpeedSignState = "(Max Speed)"
                    MaxSpeedHUD = Max50Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    SpeedSignState = "(Min Speed)"
                    MinSpeedHUD = Min50Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            elif class_name[classid[0]] == "speed sign 60mph":
                if RedPixels > BluePixels:
                    SpeedSignState = "(Max Speed)"
                    MaxSpeedHUD = Max60Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    SpeedSignState = "(Min Speed)"
                    MinSpeedHUD = Min60Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            elif class_name[classid[0]] == "speed sign 70mph":
                if RedPixels > BluePixels:
                    SpeedSignState = "(Max Speed)"
                    MaxSpeedHUD = Max70Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    SpeedSignState = "(Min Speed)"
                    MinSpeedHUD = Min70Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            elif class_name[classid[0]] == "speed sign 80mph":
                if RedPixels > BluePixels:
                    SpeedSignState = "(Max Speed)"
                    MaxSpeedHUD = Max80Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    SpeedSignState = "(Min Speed)"
                    MinSpeedHUD = Min80Message
                    cv.rectangle(b, (box[0],box[1]), (box[2],box[3]), color, 1)
                    cv.putText(b, class_name[classid[0]] + ' ' + SpeedSignState + ' %.2f' % score, (box[0], box[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            b = cvzone.overlayPNG(b, MinSpeedHUD, [0,667])
            b = cvzone.overlayPNG(b, MaxSpeedHUD, [792,667])        
        else:
            cv.rectangle(b, box, color, 1)
            cv.putText(b, label, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
    endingTime = time.time() - starting_time
    fps = math.ceil(frame_counter/endingTime)
    # print(fps)
    cv.putText(b, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('frame', b)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
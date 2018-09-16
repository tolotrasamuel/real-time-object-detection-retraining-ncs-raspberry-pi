import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/tolotra/caffe/'
sys.path.insert(0, caffe_root + 'python')  
sys.path.insert(0,  '/usr/local/cuda/liba64')

import argparse  

net_file= 'example/MobileNetSSD_deploy.prototxt'  
caffe_model='example/MobileNetSSD_deploy_4000.caffemodel'  
test_dir = "/home/tolotra/MyDataset/CustomDataset/Images"
vid_dir = "/home/tolotra/MyDataset/CustomDataset/Videos/myvid.mp4"


parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", default=vid_dir,help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--dtype", default="images",  help="Choose test data type")
args = parser.parse_args()


video = False

cap = capture =cv2.VideoCapture(args.video)
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()



#Not working from here to end

if args.dtype=="video":
    print(args.video)
    cap = cv2.VideoCapture(args.video)
    video = True
else:
    cap = cv2.VideoCapture(0)

if(video):
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (ret):
		cv2.imshow("SSD",frame)      
        	 


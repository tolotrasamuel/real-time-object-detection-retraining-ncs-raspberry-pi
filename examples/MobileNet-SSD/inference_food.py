import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/tolotra/caffe/'
sys.path.insert(0, caffe_root + 'python')  
sys.path.insert(0,  '/usr/local/cuda/liba64')

import caffe
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



if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'coke', 'milk', 'hand')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(origimg):
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)
    if(args.dtype=="video"):
	    if cv2.waitKey(1) >= 0:  # Break with ESC 
		return False
    else:
	    k = cv2.waitKey(0) & 0xff
		#Exit if ESC pressed
	    if k == 27 : return False
    return True

video = False

if args.dtype=="video":
   cap = capture =cv2.VideoCapture(args.video)
   while(cap.isOpened()):
       ret, frame = cap.read()
       if detect(frame) == False:
          break 
       cv2.waitKey(1)
   cap.release()
   cv2.destroyAllWindows()  	

elif (args.dtype=="webcame"):
    cap = cv2.VideoCapture(0)
else:
	for f in os.listdir(test_dir):
            imgfile = test_dir + "/" + f
	    if detect(origimg = cv2.imread(imgfile)) == False:
	       break

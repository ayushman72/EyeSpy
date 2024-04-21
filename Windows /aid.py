from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from collections import Counter
import torch 
import json
import requests
import pyttsx3
from multiprocessing import Process,Manager
# import serial
import time
# import string
# import pynmea2

#The GPS part of our product has not been implemented for Windows yet. Hence it gives static values.

labels={0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

def say(engine,text):
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    
def upload(data):
    while True:
        # ser=serial.Serial("/dev/serial0",baudrate = 9600,timeout = 1)
        # newdata=ser.readline()
        # if "$GPRMC" in str(newdata):
        #         newmsg = pynmea2.parse(newdata.decode("UTF-8"))
        #         lat= newmsg.latitude
        #         lng = newmsg.longitude
        #         if lat=0 or lat is None :
        #             lat,lng=(23.3173908,85.3754966)
        data["lat"]=23.3173908
        data["lng"]=85.3754966
        requests.put("http://192.168.45.230:5000/video",json.dumps(dict(data)),headers = {"Content-Type": "application/json"})
        
    
def predict(data):
    
    engine = pyttsx3.init()
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 160)
    say(engine,"EyeSpy Welcomes You")
    
    model=YOLO("yolov8n.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    say(engine,"The Model has been loaded. EyeSpy is now ready to serve you.")
    
    cap=cv2.VideoCapture(1)
    while cap.isOpened():
        _, vid = cap.read()
        if vid is not None:
            frame = vid
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            o=((output-np.min(output))/(np.max(output)-np.min(output)))
            overlay=cv2.merge([o,o,o])
            feed = (frame*overlay).astype(np.uint8)

            results=model(feed)
            for result in results:
                boxes=result.boxes.xyxy.numpy()
                marks=result.boxes.cls.numpy()


            str=""
            for k,v in Counter(result.boxes.cls.numpy()).items():
                str+=f"{v} {labels[k]} " if v>3 else f"group of {labels[k]} "
            say(engine,str) 
            data["image"]=frame.tolist()
            data["box"]=boxes.tolist() 
            data["label"]=marks.tolist() 
            
            cv2.imshow("feed", feed)
            if (cv2.waitKey(1) & 0xFF == ord("q")) or cv2.waitKey(1)==27: #27 is the escape key
                say(engine,"EyeSpy is Shutting Down")
                break        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
 
    data=Manager().dict()
    data["image"]=None
    data["box"]=None
    data["label"]=None
    data["lat"]=None
    data["lng"]=None
    p1=Process(target=predict,args=(data,),daemon=True)
    p2=Process(target=upload,args=(data,),daemon=True)
    p1.start()
    time.sleep(20)
    p2.start()
    p1.join()
    p2.join()

<h1>EyeSpy</h1>
Project EyeSpy aims to Provide an aid to people with varying degrees of visual impairments from complete blindness to partial occlusion. Our Project includes two deep learning models that are used to detect objects in close proximity of the user through a camera feed and notify the user about the type of the object through a speaker. In addition to this the inferred feed from the camera is being transferred to a web page, that can be used by a family member to monitor the user remotely and proactively take actions to mitigate dangerous situations. A GPS module is also used to send the realtime location of the user to the webpage.
<h2>Models used</h2>
Our project uses two distinct models; The MiDaS monocular Depth perception model by Intel - ISL, and the yolov8 model trained on the COCO2014 dataset. For a person that uses our model, objects that are far away would not be of much significance. The MiDaS model takes in the camera feed and provides a mask where the proximal objects are bright white, and the distant objects are grey-black. This mask is overlayed on the frame, removing the background. This new frame is then sent to the yolov8 model for object detection. Once the inference is completed, the detected objects are relayed to the user through the speaker. The inferred image is relayed to the web page in form of a numpy array.
<h2>Hardware used and inference results</h2>
To keep the device user's module independent of network limitations, we decided to go ahead with edge device inference. To this effect we used a Raspberry Pi 4 Model B, a camera module, and a GPS Neo - 6M. The inference time on the raspberry pi was 3200-4000 ms, after using multiprocessing, which included processing of the inference models and sending the data to the web page. 
The basic inference and testing was done on windows laptop with following specifications:
Windows - 10
Python 3.11.6
RAM - 8 GB
CPU - AMD Ryzen 3 5300U

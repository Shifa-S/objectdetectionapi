#camera.py
# import the necessary packages
import cv2
import numpy as np
# defining face detector
#face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
#ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
        net = cv2.dnn.readNet("C:\shifa\sem6\Internship\maskdet\maskapp\yolov3.weights", "C:\shifa\sem6\Internship\maskdet\maskapp\yolov3.cfg")
        # In[140]:
        classes = []
        # In[141]:
        with open("C:\shifa\sem6\Internship\maskdet\maskapp\coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]  
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # In[142]:
        class_ids = []
        confidences = []
        boxes = []


        # In[ ]:
        while True:
            re,img = self.video.read()
            #print(re)
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),  swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                    
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                    
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            for i in range(len(boxes)):
                if i in indexes:
                    x, y , w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if label=='person':
                        pass #VideoCamera().mask()
                    if label=='traffic light':
                        pass   # traffic()
                    c=str(confidences[i])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y -10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
    def mask(self): 
        #print("hi")
        net_mask = cv2.dnn.readNet("C:\shifa\sem6\Internship\maskdet\maskapp\yolov3_mask_last.weights", "C:\shifa\sem6\Internship\maskdet\maskapp\yolov3_mask.cfg")
        classes_mask = []
        with open("C:\shifa\sem6\Internship\maskdet\maskapp\coco -1.names", "r") as f:
            classes_mask = [line.strip() for line in f.readlines()]  
            layer_names_mask = net_mask.getLayerNames()
            output_layer_mask = [layer_names_mask[i[0] - 1] for i in net_mask.getUnconnectedOutLayers()]
            while True:
                        
                re,img = self.video.read()
                img = cv2.resize(img, None, fx=0.4, fy=0.4)
                height, width, channels = img.shape
                blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),  swapRB=True, crop=False)
                net_mask.setInput(blob)
                outs = net_mask.forward(output_layer_mask)
                class_ids_mask = []
                confidences_mask = []
                boxes_mask = []     
                for out in outs:
                    for detection in out:
                        scores_mask = detection[5:]
                        class_id_mask = np.argmax(scores_mask)
                        confidence_mask = scores_mask[class_id_mask]
                        if confidence_mask > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            
    # In[143]:              
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes_mask.append([x, y, w, h])
                            confidences_mask.append(float(confidence_mask))
                            class_ids_mask.append(class_id_mask)                
                indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(classes_mask), 3))
                for i in range(len(boxes_mask)):
                    if i in indexes_mask:
                        x, y, w, h = boxes_mask[i]
                        label_mask = str(classes_mask[class_ids_mask[i]])
                        if(label_mask=='Mask weared partially'):
                            label_mask='No mask'
    
                        c=str(confidences_mask[i])
                        color = colors[class_ids_mask[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label_mask, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                ret, jpeg = cv2.imencode('.jpg', img)
                return jpeg.tobytes()
                    
                        
            
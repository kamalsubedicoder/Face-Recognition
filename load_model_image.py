import numpy as np
import cv2
import os

import faceRecognition as fr
print (fr)


#test_img=cv2.imread(r'C:\Users\Kieron subedi\Image\Test images\success05.jpg')      #Give path to the image which you want to test
pth=str(input("Enter test image path: "))
test_img=cv2.imread(pth)

faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)


face_recognizer=cv2.face.LBPHFaceRecognizer_create()#makes histogram
face_recognizer.read(r'C:\Users\Kieron subedi\Image\trainingData.yml')  #Give path of where trainingData.yml is saved

name={0:"obama",1:"michelle",2:"Trump",3:"Jennifer",4:"Brad",5:"Anne",6:"Angelina" }

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>50):
        fr.put_text(test_img,'unknown',x,y);
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(350,450))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

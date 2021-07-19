import numpy as np
import cv2
import os

import faceRecognition as fr




#Training will begin from here

faces,faceID=fr.labels_for_training_data(r'C:\Users\Kieron subedi\Image\train-images') #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:\Users\Kieron subedi\Image\trainingData.yml') #It will save the trained model. 
print("Training with LBPH algorithm complete. ")

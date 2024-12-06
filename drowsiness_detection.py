#importing all the dependencies 

import cv2
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Loading Alarm Sound
mixer.init()

sound = mixer.Sound(r"C:\Users\GREESHMA\Desktop\Drowsines_detection\Drowsines_detection\alarm.wav")

# Loading Haar Cascade for face, left and right eye
face_cascade = cv2.CascadeClassifier(r'C:\\Users\\GREESHMA\\Desktop\\Drowsines_detection\\Drowsines_detection\\haar cascade files\\haarcascade_frontalface_alt.xml')
eye_cascade_left = cv2.CascadeClassifier(r'C:\\Users\\GREESHMA\\Desktop\\Drowsines_detection\\Drowsines_detection\\haar cascade files\\haarcascade_lefteye_2splits.xml')
eye_cascade_right = cv2.CascadeClassifier(r'C:\\Users\\GREESHMA\\Desktop\\Drowsines_detection\\Drowsines_detection\\haar cascade files\\haarcascade_righteye_2splits.xml')



#define label 

lbl=['Close','Open']
import tensorflow as tf

model = tf.keras.models.load_model(r'C:\\Users\\GREESHMA\\Desktop\\Drowsines_detection\\Drowsines_detection\\models\\Trained file.h5')
# model = load_model(r'C:\Users\GREESHMA\Desktop\Drowsines_detection\Drowsines_detection\models\Trained file.h5')
#load the model
path = os.getcwd()          #Get Current working directory
cap = cv2.VideoCapture(0)           #Initialize Video capture
font = cv2.FONT_HERSHEY_COMPLEX_SMALL          #Define font for overlay
# Initializing variables
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

# Start the main loop
while(True):
    ret, frame = cap.read()    #Read the video frame
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #covert the frame to grayscale
    
    #faces = frame.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    faces = face_cascade.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))                #Detecting faces,left eyes, and right eyes in the frame

    # Loading Haar Cascade for face, left and right eye
    face_cascade = cv2.CascadeClassifier(r'C:\\Users\\GREESHMA\\Desktop\\Drowsines_detection\\Drowsines_detection\\haar cascade files\\haarcascade_frontalface_alt.xml')
    eye_cascade_left = cv2.CascadeClassifier(r'C:\\Users\\GREESHMA\\Desktop\\Drowsines_detection\\Drowsines_detection\\haar cascade files\\haarcascade_lefteye_2splits.xml')
    eye_cascade_right = cv2.CascadeClassifier(r'C:\\Users\\GREESHMA\\Desktop\\Drowsines_detection\\Drowsines_detection\\haar cascade files\\haarcascade_righteye_2splits.xml')

# ...

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = eye_cascade_left.detectMultiScale(gray)
    right_eye = eye_cascade_right.detectMultiScale(gray)
    #left_eye = leye.detectMultiScale(gray)
    #right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )       #Drawing a black rectanlge at the bottom of the frame for displaying text which will be open/closed with score

    for (x,y,w,h) in faces:                                                                     #Process each detected face
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:                                                      #Process the right eye
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye),axis=1) 
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:                                                        #Process the Left eye
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye),axis=1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):                                                            #If the eyes remain closed the score will increment by 1
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)                   # else open eyes the score will decrease by -1
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)                           #will put the score in the black border below
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

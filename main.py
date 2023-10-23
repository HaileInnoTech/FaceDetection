import pickle
import numpy as np
import cv2
import os
import face_recognition
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgBackground = cv2.imread('Resources/background.png')

#Import mode to to a List
folderModePath = "Resources/Modes"
modePathList = os.listdir(folderModePath)
imageModeList =[]
for path in modePathList:
    imageModeList.append(cv2.imread(os.path.join(folderModePath,path)))
# Load the encoding file
print("Loading Encode File ...")
file = open('EncoderFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown,peopleIds = encodeListKnownWithIds
# print(peopleIds)
print("Encoded File Loaded")

while True:
    ret, img = cap.read()
    imgS = cv2.resize(img, (0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    enCodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imageModeList[0]

    for encodeFace, faceLoc in zip(enCodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print("matches" , matches)
        print("Distance" , faceDis)

        matchIndex = np.argmin(faceDis)
        print("Match Index" , matchIndex)
        if  matches[matchIndex]:
            print("Face Detected")
            print(peopleIds[matchIndex])
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            bbox = 55+x1,162+y1, x2-x1, y2-y1
            cvzone.cornerRect(imgBackground,bbox, rt=0)




    cv2.imshow("Face Attendence", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()

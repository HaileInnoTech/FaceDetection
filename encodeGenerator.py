import cv2
import face_recognition
import os
import pickle

# Import people images
folderPath = "Images"
pathList = os.listdir(folderPath)
imageList =[]
peopleIds=[]
for path in pathList:
    imageList.append(cv2.imread(os.path.join(folderPath,path)))
    peopleIds.append(os.path.splitext(path)[0])
print(peopleIds)

def FindEncodings(imageList):
    encodeList =[]
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        return encodeList

print("Encoding Started ....")
encodeListKnown = FindEncodings(imageList)
encodeListKnownWithIds = [encodeListKnown,peopleIds]
print("Encoding Completed ")

file = open("EncoderFile.p", 'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")

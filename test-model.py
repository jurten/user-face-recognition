import cv2

# Determines what webcam will it use, 0 for default
video=cv2.VideoCapture(0)

# Loads the classifier
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainingData.yml")

names = ['', 'Joaco']

while True:
    ret,frame=video.read()

    # Converts the frame to gray
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detects the face
    faces = face_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    # creates a rectangle around the face
    for x,y,w,h in faces:
        serial, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if confidence > 60:
            cv2.rectangle(frame, (x,y-40), (x+w,y), (0,255,0), -1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # 2 is the thickness of the rectangle
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.putText(frame,names[serial],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        else:
            cv2.rectangle(frame, (x,y-40), (x+w,y), (0,0,255), -1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) # 2 is the thickness of the rectangle
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(frame,"Unknown",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame",frame)
    cv2.waitKey(1)

    keylistener = cv2.waitKey(1)

    if keylistener == ord('q'):
        break

video.release()
cv2.destroyAllWindows()




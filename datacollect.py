import cv2

# Determines what webcam will it use, 0 for default
video=cv2.VideoCapture(0)

# Loads the classifier
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter user ID: ")
count = 0

while count<500:
    ret,frame=video.read()

    # Converts the frame to gray
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detects the face
    faces = face_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    # creates a rectangle around the face
    for x,y,w,h in faces:
        count+=1
        cv2.imwrite("datasets/user."+id+"."+str(count)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # 2 is the thickness of the rectangle

    cv2.imshow("Frame",frame)

    key_listener=cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()
print("Data collection complete")




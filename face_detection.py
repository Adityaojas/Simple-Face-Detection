import cv2
import imutils

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width = 1024, height = 1024)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy() 
    
    rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=5,
                                     minSize=(70,70), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (fX, fY, fW, fH) in rects:
        cv2.rectangle(frame_clone, (fX,fY), (fX+fW, fY+fH), (0,255,0), 2)
        cv2.imshow("Face", frame_clone)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
camera.release()
cv2.destroyAllWindows
        


import cv2

cap = cv2.VideoCapture(0)

attemptOpenCam = 0
while(not cap.isOpened()):
   attemptOpenCam += 1
   if(attemptOpenCam == 1):
      print('Camera is not detected yet, please be patient, or check cam')
   cap.release()
   cv2.detroyAllWindows()
   cap = cv2.VideoCapture(0)
print('Camera is on!')

while(True):
   ret, frame = cap.read()

   # Display the resulting image
   cv2.imshow('Video', frame)

   print('When ready, take a photo by hitting t')

   if cv2.waitKey(1) & 0xFF == ord('t'):
        ret, frame = cap.read()
        cv2.imwrite('tianxiang4.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()

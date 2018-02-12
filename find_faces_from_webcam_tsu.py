import face_recognition
import cv2

# Open webcam
cap = cv2.VideoCapture(0)

attemptOpenCam = 0
while(not cap.isOpened()):
    attemptOpenCam += 1
    if(attemptOpenCam == 1):
       print("Camera not detected yet, please be patient")
    cap.release()
    cv2.destroyAllWindows()
    cap = cv2.captureVideo(0)
print("Camera is on!")


numFrameInGroup = 25
frameID = -1
while(True):
    frameID += 1

    # image = face_recognition.load_image_file("tianxiang.jpeg")
    # print image.shape

    ret, image = cap.read()

    if(frameID % numFrameInGroup == 0):
       # Find all the faces in the image using the default HOG-based model.
       # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
       # See also: find_faces_in_picture_cnn.py
       face_locations = face_recognition.face_locations(image)
       # face_locations = face_recognition.face_locations(image, model="cnn")
       # print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

       # Print the location of each face in this image
       # top, right, bottom, left = face_location
       # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

       # You can access the actual face itself like this:
       # face_image = image[top:bottom, left:right]
       # pil_image = Image.fromarray(face_image)
       # pil_image.show()

       top, right, bottom, left = face_location
       # Scale back up face locations since the frame we detected in was scaled to 1/4 size
       # top *= 4
       # right *= 4
       # bottom *= 4
       # left *= 4

       # Draw a box around the face
       cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

       # Draw a label with a name below the face
       # cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
       # font = cv2.FONT_HERSHEY_DUPLEX
       # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Faces', image)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()   
cv2.waitKey()

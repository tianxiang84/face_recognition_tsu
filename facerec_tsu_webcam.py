import face_recognition
import cv2
import numpy as np
import glob
import sys

lucas_face_encoding = []
for filename in glob.glob('raw_imgs/*'):
   print(filename)
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   lucas_face_encoding.append(face_recognition.face_encodings(image,face_locations)[0])

# Create arrays of known face encodings and their names
known_face_encodings = [
    lucas_face_encoding
]

known_face_names = [
    "Lucas"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

font = cv2.FONT_HERSHEY_DUPLEX

while True:
    # Grab a single frame of video
    frame = cv2.imread('test/2019-03-16 164209.jpg')

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if True:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
        # See how far apart the test image is from the known faces
        face_names = []
        for face_encoding in face_encodings:
            face_distances = []
            name = "Please Get Closer...."
            for known_face_encoding in known_face_encodings:
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                face_distances.append( reduce(lambda x, y: x+y, face_distance) / len(face_distance) )
            index = np.argmin(face_distances)
            dist = face_distances[index]
            
            if(dist<0.5):
               name = known_face_names[index]
            elif(dist<0.6):
               name = known_face_names[index] + "?"
            
            face_names.append(name)
            


        # Find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
    
    
    
    # Display the results
    for face_landmarks in face_landmarks_list:
        # Print the location of each facial feature in this image
        facial_features = [
              'chin',
              'left_eyebrow',
              'right_eyebrow',
              'nose_bridge',
              'nose_tip',
              'left_eye',
              'right_eye',
              'top_lip',
              'bottom_lip'
         ]

        for facial_feature in facial_features:
            for pts in face_landmarks[facial_feature]:
                cv2.circle(frame, pts, 2, (0,255,0), 1)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
#        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
#        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #cv2.putText(frame, name, (left - 29, bottom - 76), font, 0.6, (255, 255, 255), 1)

        cv2.rectangle(frame, (left-35, top + 70), (right+35, bottom-70), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left-35, bottom - 105), (right+35, bottom-70), (0, 0, 255), cv2.FILLED)
 
        cv2.putText(frame, name, (left - 29, bottom - 76), font, 0.6, (0, 0, 0), 1)



    # Display the resulting image
    cv2.imshow('Video',frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

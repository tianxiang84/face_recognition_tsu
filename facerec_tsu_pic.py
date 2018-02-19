import face_recognition
import cv2
import numpy as np
import glob
import sys

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.



# Create Library
tianxiang_face_encoding = []
for filename in glob.glob('tianxiang/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   tianxiang_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#ying_face_encoding = []
#for filename in glob.glob('ying/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   ying_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#meng_face_encoding = []
#for filename in glob.glob('meng/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   meng_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#xianjun_face_encoding = []
#for filename in glob.glob('xianjun/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   xianjun_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

muhannad_face_encoding = []
for filename in glob.glob('muhannad/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   muhannad_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

jeremy_face_encoding = []
for filename in glob.glob('jeremy/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   jeremy_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#miranda_face_encoding = []
#for filename in glob.glob('miranda/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   miranda_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

sepand_face_encoding = []
for filename in glob.glob('sepand/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   sepand_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# anastasia_face_encoding = []
# for filename in glob.glob('anastasia/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   anastasia_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# jan_face_encoding = []
# for filename in glob.glob('jan/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   jan_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# demos_face_encoding = []
# for filename in glob.glob('demos/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   demos_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# cecilia_face_encoding = []
# for filename in glob.glob('cecilia/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   cecilia_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# chris_face_encoding = []
# for filename in glob.glob('chris/*'):
#    image = cv2.imread(filename)
#    face_locations = face_recognition.face_locations(image)
#    chris_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

gavin_face_encoding = []
for filename in glob.glob('gavin/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   gavin_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

matt_face_encoding = []
for filename in glob.glob('matt/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   matt_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

rich_face_encoding = []
for filename in glob.glob('rich/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   rich_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

rod_face_encoding = []
for filename in glob.glob('rod/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   rod_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#alejandro_face_encoding = []
#for filename in glob.glob('alejandro/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   alejandro_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#nicolas_face_encoding = []
#for filename in glob.glob('nicolas/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   nicolas_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#walid_face_encoding = []
#for filename in glob.glob('walid/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   walid_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])


# Create arrays of known face encodings and their names
known_face_encodings = [
    tianxiang_face_encoding,
    #ying_face_encoding,
    #meng_face_encoding,
    #xianjun_face_encoding,
    muhannad_face_encoding,
    jeremy_face_encoding,
    #miranda_face_encoding,
    sepand_face_encoding,
    #anastasia_face_encoding,
    #alejandro_face_encoding,
    #nicolas_face_encoding,
    #walid_face_encoding,
    #jan_face_encoding,
    #demos_face_encoding,
    #cecilia_face_encoding,
    #chris_face_encoding,
    gavin_face_encoding,
    matt_face_encoding,
    rich_face_encoding,
    rod_face_encoding
]

known_face_names = [
    "Tianxiang",
    #"Ying",
    #"Meng",
    #"Xianjun",
    "Muhannad",
    "Jeremy",
    #"Miranda",
    "Sepand",
    #"Anastasia",
    #"Alejandro",
    #"Nicolas",
    #"Walid",
    #"Jan",
    #"Demos",
    #"Cecilia",
    #"Chris",
    "Gavin",
    "Matt",
    "Rich",
    "Rod"
]




frame = cv2.imread('tianxiang/tianxiang2.jpg')

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

#process_this_frame = True
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video',640,480)
font = cv2.FONT_HERSHEY_DUPLEX


# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
# rgb_small_frame = small_frame[:, :, ::-1]
rgb_small_frame = frame[:, :, ::-1]

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
            cv2.circle(frame, pts, 2, (0,0,255), 2)

# Display the results
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
cv2.waitKey(0)

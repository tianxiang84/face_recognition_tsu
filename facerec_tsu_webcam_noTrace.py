import face_recognition
import cv2
import numpy as np
import glob

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.



# Open Camera
PORT = 0 
# PORT = 'http://100.106.104.82:4747/mjpegfeed'
# video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture('http://10.0.0.173:4747/mjpegfeed')
video_capture = cv2.VideoCapture(PORT)

attemptOpenCam = 0
while(not video_capture.isOpened()):
   attemptOpenCam += 1
   if(attemptOpenCam == 1):
      print('Camera not detected yet, please be patient...')
   video_capture.release()
   cv2.destroyAllWindows()
   video_capture = cv2.VideoCapture(PORT)
print('Camera is on!')


## Create Library
# image = cv2.imread("tianxiang/tianxiang.jpeg")
# face_locations = face_recognition.face_locations(image)
# tianxiang_face_encoding = face_recognition.face_encodings(image, face_locations)[0]
# print tianxiang_face_encoding.shape

tianxiang_face_encoding = []
for filename in glob.glob('tianxiang/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   tianxiang_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# image = cv2.imread("ying/ying.jpg")
# face_locations = face_recognition.face_locations(image)
# ying_face_encoding = face_recognition.face_encodings(image, face_locations)[0]

#ying_face_encoding = []
#for filename in glob.glob('ying/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   ying_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# image = cv2.imread("meng/meng.jpg")
# face_locations = face_recognition.face_locations(image)
# meng_face_encoding = face_recognition.face_encodings(image, face_locations)[0]

#meng_face_encoding = []
#for filename in glob.glob('meng/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   meng_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])


# image = cv2.imread("xianjun/xianjun.jpg")
# face_locations = face_recognition.face_locations(image)
# xianjun_face_encoding = face_recognition.face_encodings(image, face_locations)[0]

#xianjun_face_encoding = []
#for filename in glob.glob('xianjun/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   xianjun_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])


# image = cv2.imread("muhannad/muhannad.jpg")
# face_locations = face_recognition.face_locations(image)
# muhannad_face_encoding = face_recognition.face_encodings(image, face_locations)[0]

muhannad_face_encoding = []
for filename in glob.glob('muhannad/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   muhannad_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# image = cv2.imread("jeremy/jeremy2.jpg")
# face_locations = face_recognition.face_locations(image)
# jeremy_face_encoding = face_recognition.face_encodings(image, face_locations)[0]

jeremy_face_encoding = []
for filename in glob.glob('jeremy/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   jeremy_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

# image = cv2.imread("miranda/miranda.jpg")
# face_locations = face_recognition.face_locations(image)
# miranda_face_encoding = face_recognition.face_encodings(image, face_locations)[0]

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

chris_face_encoding = []
for filename in glob.glob('chris/*'):
    image = cv2.imread(filename)
    face_locations = face_recognition.face_locations(image)
    chris_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

gavin_face_encoding = []
for filename in glob.glob('gavin/*'):
   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   gavin_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#matt_face_encoding = []
#for filename in glob.glob('matt/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   matt_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#rich_face_encoding = []
#for filename in glob.glob('rich/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   rich_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

#rod_face_encoding = []
#for filename in glob.glob('rod/*'):
#   image = cv2.imread(filename)
#   face_locations = face_recognition.face_locations(image)
#   rod_face_encoding.append(face_recognition.face_encodings(image, face_locations)[0])

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
    chris_face_encoding,
    gavin_face_encoding,
    #matt_face_encoding,
    #rich_face_encoding,
    #rod_face_encoding
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
    "Chris",
    "Gavin",
    #"Matt",
    #"Rich",
    #"Rod"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

#process_this_frame = True
numInGroup = 10
frameID = -1
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video',640,480)
font = cv2.FONT_HERSHEY_DUPLEX

while True:
    frameID += 1

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
#    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#    small_frame = frame

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#    rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = frame[:, :, ::-1]

    # Only process every other frame of video to save time
#    if process_this_frame:
    if(frameID % numInGroup == 0):
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
               # print(index)
               name = known_face_names[index]
            elif(dist<0.6):
               # print(index)
               name = known_face_names[index] + "?"
            
            # print(face_distances)
            # print(name, face_distances[index])
            face_names.append(name)
            

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left, bottom), font, 0.9, (255, 255, 255), 1)

        #cv2.rectangle(frame, (left-35, top + 70), (right+35, bottom-70), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left-35, bottom - 105), (right+35, bottom-70), (0, 0, 255), cv2.FILLED)
 
        #cv2.putText(frame, name, (left - 29, bottom - 76), font, 0.6, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Video',frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

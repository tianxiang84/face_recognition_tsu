import face_recognition
import cv2
import numpy as np
import glob
import sys
import functools

# 1. Prepare
# 1.1 Prepare ML
# 1.2 Prepare vid output

# 2. Loop over images
# 2.1 Detect Lucas, get coordinate
# 2.2 Detect Lucas' eyes, get coordinates
# 2.3 Rotate the image
# 2.4 Determine the 4 corners of the images
# 2.5 Put the image
# 2.6 output vid


# 1.1 Prepare ML
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

# 1.2 Prepare video output
vid_out = cv2.VideoWriter('aligned_vid.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (3264,2448))
#sys.exit()


# Initialize some variables
#face_locations = []
#face_encodings = []
#face_names = []
#font = cv2.FONT_HERSHEY_DUPLEX


for filename in glob.glob('test/*'):
    print(filename)

    # Grab a single frame of video
    frame = cv2.imread(filename)
    h = frame.shape[0]
    w = frame.shape[1]
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = frame[:, :, ::-1]


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_distances = []
    for face_encoding in face_encodings:
        face_distance = face_recognition.face_distance(known_face_encodings[0], face_encoding)
        face_distances.append( functools.reduce(lambda x, y: x+y, face_distance) / len(face_distance) )
    index = np.argmin(face_distances)
    #dist = face_distances[index]


    (top, right, bottom, left) = face_locations[index]

    LEFT = int(5000 - (left+right)/2.0)
    #RIGHT = 5000 - (left+right)/2.0 + w
    TOP = int(5000 - (top+bottom)/2.0)
    #BOTTOM = 5000 - (top+bottom)/2.0 + h

    canvas = np.zeros([10000,10000,3])
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    canvas[TOP:TOP+h, LEFT:LEFT+w, :] = frame

    cv2.imwrite(filename.split('.')[0]+'_detected.jpg', canvas)

    print('Done')

    '''
    print('showing')
    # Display the resulting image
    cv2.imshow('Video',frame)
    cv2.waitKey()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print('exit')
    #sys.exit()
    '''

    '''
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
    '''

vid_out.release()
#cv2.destroyAllWindows()

import face_recognition
import cv2
import numpy as np
import glob
import sys
import functools
import os
import imutils

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
## Prepare face encoding for Lucas
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
vid_out = cv2.VideoWriter('aligned_vid.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 1.0, (1000,1000))


# Initialize some variables
#face_locations = []
#face_encodings = []
#face_names = []
#font = cv2.FONT_HERSHEY_DUPLEX

file_list = glob.glob('test/*.jpg')
for filename in file_list:
    print('Processing {}'.format(filename))

    # Input the photo and get its height and width
    frame = cv2.imread(filename)
    h = frame.shape[0]
    w = frame.shape[1]
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_distances = []
    for face_encoding in face_encodings:
        face_distance = face_recognition.face_distance(known_face_encodings[0], face_encoding)
        face_distances.append(functools.reduce(lambda x, y: x+y, face_distance) / len(face_distance))
    index = np.argmin(face_distances)
    #dist = face_distances[index]

    # location 
    (top, right, bottom, left) = face_locations[index]
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)




    # Find all facial features in all the faces in the image
    dist = 0
    ang = 0
    mid_eye_x = 0
    mid_eye_y = 0
 
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

    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    for face_landmarks in face_landmarks_list:
        # Print the location of each facial feature in this image
        if not face_landmarks['left_eyebrow'][0][0]>=left:
               continue
        if not face_landmarks['left_eyebrow'][0][0]<=right:
               continue
        if not face_landmarks['left_eyebrow'][0][1]>=top:
               continue
        if not face_landmarks['left_eyebrow'][0][1]<=bottom:
               continue

        for facial_feature in facial_features:
            for pts in face_landmarks[facial_feature]:
                cv2.circle(frame, pts, 2, (0,255,0), 1)

        lx = 0
        ly = 0
        n_pt = 0
        for coordinates in face_landmarks['left_eyebrow']:
            lx = lx + coordinates[0]
            ly = ly + coordinates[1]
            n_pt = n_pt + 1
        lx = lx / n_pt
        ly = ly / n_pt

        rx = 0
        ry = 0
        n_pt = 0
        for coordinates in face_landmarks['right_eyebrow']:
            rx = rx + coordinates[0]
            ry = ry + coordinates[1]
            n_pt = n_pt + 1
        rx = rx / n_pt
        ry = ry / n_pt

        mid_eye_x = (lx+rx)/2.0
        mid_eye_y = (ly+ry)/2.0
        dist = np.sqrt((lx-rx)*(lx-rx)+(ly-ry)*(ly-ry))
        ang = np.arctan2((ry-ly),(rx-lx))
        print(dist, ang*180/np.pi)


    LEFT = int(5000 - mid_eye_x)
    TOP = int(5000 - mid_eye_y)
    canvas = np.zeros([10000,10000,3])
    canvas[TOP:TOP+h, LEFT:LEFT+w, :] = frame
    canvas = imutils.rotate_bound(canvas, -ang*180.0/np.pi)

    scale = 300/dist
    new_h = int(10000*scale)
    new_w = int(10000*scale)
    print(scale)
    print(new_h)
    print(new_w)
    print(h)
    print(w)

    dx = int((new_w-1000)/2.0)
    dy = int((new_h-1000)/2.0)
    canvas = cv2.resize(canvas,(new_w, new_h))
    canvas = canvas[dy:dy+1000, dx:dx+1000, :]

    fn = filename.split('.')[0]
    fn = fn.split('/')[1]
    cv2.imwrite(os.path.join('test','result',fn+'_detected.jpg'), canvas)
    print(canvas.shape)

    vid_out.write(np.uint8(canvas))

    print('Done')
vid_out.release()
#cv2.destroyAllWindows()

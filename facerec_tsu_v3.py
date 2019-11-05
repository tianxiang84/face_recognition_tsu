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
for idx, filename in enumerate(glob.glob('imgs_for_training/*.jpg')):
   if idx%10 == 0:
       print('File {}: {}'.format(idx, filename))

   image = cv2.imread(filename)
   face_locations = face_recognition.face_locations(image)
   if len(face_locations)==0:
       print('Training no face found: {}'.format(filename))
       continue
   lucas_face_encoding.append(face_recognition.face_encodings(image,face_locations)[0])   
   #print(len(lucas_face_encoding[-1]))
print('Total {} faces encoded.'.format(len(lucas_face_encoding)))

# Create arrays of known face encodings and their names
known_face_encodings = [lucas_face_encoding]
known_face_names = ["Lucas"]

# 1.2 Prepare video output
OUTPUT_VID_H = 250
OUTPUT_VID_W = 250
vid_out = cv2.VideoWriter('aligned_vid_v2.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 5.0, (OUTPUT_VID_H,OUTPUT_VID_W))



# Get the list of images
folders_name = ['2018_12','2019_01','2019_02','2019_03','2019_04','2019_05','2019_06','2019_07','2019_08','2019_09','2019_10','2019_11','2019_12']
images_list = []
for folder_name in folders_name:
    this_file_list = glob.glob(os.path.join('imgs_for_vid', folder_name, '*.jpg'))
    this_file_list.sort()
    images_list = images_list + this_file_list



# Create a mask for blurring
mask = np.zeros([OUTPUT_VID_H, OUTPUT_VID_W])
MASK_R = 80
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if np.sqrt((i-OUTPUT_VID_H/2.0)*(i-OUTPUT_VID_H/2.0)+(j-OUTPUT_VID_W/2.0)*(j-OUTPUT_VID_W/2.0))<MASK_R:
            mask[i][j]=1.0


# Loop over all file
for filename in images_list:
    print('Processing {}'.format(filename))

    # Get the text to be written on the image
    folder_name = os.path.dirname(filename)
    folder_name = folder_name.split('/')
    folder_name = folder_name[1]
    #print(folder_name)
    text = ''
    if folder_name == '2018_12':
        text = ['2018-','12']
    if folder_name == '2019_01':
        text = ['2019-','01']
    if folder_name == '2019_02':
        text = ['2019-','02']
    if folder_name == '2019_03':
        text = ['2019-','03']
    if folder_name == '2019_04':
        text = ['2019-','04']
    if folder_name == '2019_05':
        text = ['2019-','05']
    if folder_name == '2019_06':
        text = ['2019-','06']
    if folder_name == '2019_07':
        text = ['2019-','07']
    if folder_name == '2019_08':
        text = ['2019-','08']
    if folder_name == '2019_09':
        text = ['2019-','09']
    if folder_name == '2019_10':
        text = ['2019-','10']
    if folder_name == '2019_11':
        text = ['2019-','11']
    if folder_name == '2019_12':
        text = ['2019-','12']

    # Input the photo and get its height and width
    frame = cv2.imread(filename)
    h = frame.shape[0]
    w = frame.shape[1]
    #print(h)
    #print(w)
    #sys.exit()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        print('Vid face no found.')
        continue

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_distances = []
    for face_encoding in face_encodings:
        face_distance = face_recognition.face_distance(known_face_encodings[0], face_encoding)
        face_distances.append(functools.reduce(lambda x, y: x+y, face_distance) / len(face_distance))
    index = np.argmin(face_distances)
    #dist = face_distances[index]

    # location 
    (top, right, bottom, left) = face_locations[index]
    #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


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

        #for facial_feature in facial_features:
        #    for pts in face_landmarks[facial_feature]:
        #        cv2.circle(frame, pts, 2, (0,255,0), 1)

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

    if dist < 60:
        print('Face not found...')
        continue

    LEFT = int(2500 - mid_eye_x)
    TOP = int(2500 - mid_eye_y)
    canvas = np.zeros([5000,5000,3])
    canvas[:,:,1] = 255
    canvas[:,:,2] = 255
    canvas[TOP:TOP+h, LEFT:LEFT+w, :] = frame
    canvas = imutils.rotate_bound(canvas, -ang*180.0/np.pi)

    scale = 300/dist
    new_h = int(5000*scale)
    new_w = int(5000*scale)
    canvas = cv2.resize(canvas,(new_w, new_h))

    dx = int((new_w-1200)/2.0)
    dy = int((new_h-1200)/2.0)
    canvas = canvas[dy:dy+1200, dx:dx+1200, :]
    canvas = cv2.resize(canvas, (OUTPUT_VID_W, OUTPUT_VID_H))

    blurred = cv2.GaussianBlur(canvas,(9,9),cv2.BORDER_DEFAULT)
    blurred[mask==1] = canvas[mask==1]
    canvas = blurred

    fn = filename.split('.')[0]
    fn = fn.split('/')[2]
    cv2.putText(canvas,text[0],(115,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(canvas,text[1],(180,45),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),3,cv2.LINE_AA)
    cv2.imwrite(os.path.join('processed_imgs',fn+'_detected.jpg'), canvas)
    #print(canvas.shape)

    #vid_out.write(np.uint8(canvas))

    print('Done')
vid_out.release()
#cv2.destroyAllWindows()

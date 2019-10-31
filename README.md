# face_recognition_tsu
Using Adam Geitgey's library to do face recognition

docker build -t tianxiang84/opencv:latest .
docker run -it --rm -v /home/TSu/baidunetdiskdownload/face_recognition_tsu:/home/TSu/baidunetdiskdownload/facerec_recognition_tsu --env="DISPLAY" -v /tmp/.X11-unix/:/tmp/.X11-unit tianxiang84/opencv:latest /bin/bash
cd home/TSu/baidunetdiskdownload/facerec_recognition_tsu
python facerec_tsu_webcam.py

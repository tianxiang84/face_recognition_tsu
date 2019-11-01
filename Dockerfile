FROM jjanzic/docker-python3-opencv:latest

RUN apt-get update -q

RUN apt-get -y -q --no-install-recommends install vim gtk2.0 build-essential libgtk2.0-dev

RUN python3 -m pip install Pillow dlib git+https://github.com/ageitgey/face_recognition_models pkgconfig imutils

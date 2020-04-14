from PIL import Image
import face_recognition
import cv2
import os

video_capture = cv2.VideoCapture(-1)
known_face_encodings=[]
known_face_names = []
cascPath = "haarcascade.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
root = ("photo")

def lirephoto() :
    for filename in os.listdir(root):
        if filename.endswith('.jpg' ):
            try:
                print(filename)
                path = os.path.join(root, filename)
                filter_image = face_recognition.load_image_file(path)
                filter_face_encoding = face_recognition.face_encodings(filter_image)
                known_face_encodings.append(filter_face_encoding[0])
                known_face_names.append(filename)

            except:
                print("An exception occurred : " + filename )

print(known_face_names)
face_locations = []
face_encodings = []
face_names = []
global etat 
global name

def face_detect(img,name=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_flag = cv2.CASCADE_SCALE_IMAGE
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv_flag)
        for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if name is None:
                    name = 'Unknown'
                    color = (0, 0, 255)  # red for unrecognized face
                else:
                    color = (0, 128, 0)  # dark green for recognized face
                cv2.putText(img, name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

def face_reco(frame):
    etat = False
    name = " "
    while True:
        process_this_frame = True
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Inconnu"
                etat = False
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    name = name.split('.')[0]
                    etat = True
                face_names.append(name)
        face_detect(frame,name) 
        process_this_frame = not process_this_frame
        return etat , name 

lirephoto()
while True :
    _,img = video_capture.read()
    scale_percent = 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    _,name = face_reco(frame)
    cv2.imshow("frame",frame)
    k = cv2.waitKey(1)
    if k == 27 :
        break 

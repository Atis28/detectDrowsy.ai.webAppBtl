"""
tạo app desktop
"""



import socket,cv2, pickle,struct
import requests
import time
import threading
import numpy as np


client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '20.213.72.10'

port = 8000 #defaut
port = int(requests.get('http://20.213.72.10:5000/request').text)

print('HOST IP:',host_ip)
print('PORT:',port)
client_socket.connect((host_ip,port))


def get_notification(client_socket):
    while True:
        data = client_socket.recv(4096)
        message = data.decode()

        print(message)


t = threading.Thread(target=get_notification, args=(client_socket,))
t.start()

leye = cv2.CascadeClassifier('lefteye.xml')
reye = cv2.CascadeClassifier('righteye.xml')

vid = cv2.VideoCapture(0)
while(vid.isOpened()):
    try:
        img,frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        leye_pos = leye.detectMultiScale(gray)
        reye_pos = reye.detectMultiScale(gray)
        leye_frame, reye_frame = None, None

        for (x,y,w,h) in leye_pos:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            leye_frame = frame[y:y+h,x:x+w]
            break
            
        for (x,y,w,h) in reye_pos:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            reye_frame = frame[y:y+h,x:x+w]
            break

        if leye_frame is not None and reye_frame is not None:
            a = pickle.dumps(leye_frame)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)

            a = pickle.dumps(reye_frame)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)

        cv2.imshow('TRANSMITTING VIDEO',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    except KeyboardInterrupt:
        client_socket.close()
        break

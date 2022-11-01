#Khai báo các thư viện cần sử dụng
from flask import Flask, render_template, Response
import cv2
import os
from keras.models import load_model
import numpy as np
import time


app=Flask(__name__)
port = 8000

def gen_frames(frame, leye, reye, model):
    
    #Khởi tạo bộ đếm và điểm đánh giá 
    rpred=[[99] * 99]
    lpred=[[99] * 99]

    height,width = frame.shape[:2]
    #Chuyển đổi ảnh sang hệ gray 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    #Detect ra mắt phải và trái, giá trị trả về gồm có tọa độ mắt kèm theo chiều dài và chiều rộng của hình chữ nhật bao quanh mắt
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    
    
    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h,x:x+w]
        l_eye = cv2.resize(l_eye,(84,84))
        l_eye = np.array(l_eye).reshape(1, 84, 84, 3)
        l_eye = l_eye/255.0
        lpred = (model.predict(l_eye) > 0.5).astype("int32")
        break
        
    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h,x:x+w]
        r_eye = cv2.resize(r_eye,(84,84))
        r_eye = np.array(r_eye).reshape(1, 84, 84, 3)
        r_eye = r_eye/255.0
        rpred = (model.predict(r_eye) > 0.5).astype("int32")
        break
    
    if lpred[0][0] == 0 or rpred[0][0] == 0:
        return 0
    else: 
        return 1



@app.route('/request')
def request():
    import threading
    t = threading.Thread(target=new_process)
    t.start()
    global port
    # return port as a string
    return str(port)
    
def new_process():
    global port
    port += 1 # 1 port per process
    print('new process on port', port)

    import socket, cv2, pickle,struct,imutils,requests

    server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    host_name  = socket.gethostname()
    host_ip = '0.0.0.0'
    socket_address = (host_ip,port)

    server_socket.bind(socket_address)
    server_socket.listen(1) # 1 connection at a time

    client_socket,addr = server_socket.accept()
    print('GOT CONNECTION FROM:',addr)

    first_time_close = 0
    alert = False


    
    #Sử dụng bộ lọc Haar để nhận diện mắt trái
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    #Sử dụng bộ lọc Haar để nhận diện mắt phải
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

    #Load model đã train được file .h5
    model = load_model('model_opencloseeyes_best.h5')


    if client_socket:
        data = b""
        payload_size = struct.calcsize("Q")
        import time
        begin = time.time()
        while True:
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024)
                if not packet: break
                data+=packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q",packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4*1024)
            frame_data = data[:msg_size]
            data  = data[msg_size:]
            frame = pickle.loads(frame_data)   

            # sử dụng frame ở đây

            # show
            # cv2.imshow("RECEIVING VIDEO",frame)

            t = int(time.time()*10)
            if 0 <= t%10 <= 5: #chi
                alert = False
                res = gen_frames(frame, leye, reye, model)
                if res == 1: #closed eyes
                    if first_time_close == 0: # chưa từng nhắm sau khi mở mắt
                        first_time_close = time.time()
                    else:
                        if time.time() - first_time_close > 15:
                            alert = True
                else: #open eyes
                    first_time_close = 0

                #gửi dữ liệu về client
                seconds = int(time.time())
                if seconds % 2 == 0:
                    if res == 1:
                        msg = 'closed'
                    else:
                        msg = 'opened'
                    client_socket.send(msg.encode())
                    if alert:
                        msg = 'alert'
                        client_socket.send(msg.encode())
                        alert = False

        client_socket.close()



if __name__=='__main__':
    app.run()


    
#Khai báo các thư viện cần sử dụng
from flask import Flask, render_template, Response
import cv2
import cv2
import os
from keras.models import load_model
import numpy as np
import time
from pygame import mixer
import pygame

app=Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():  
    
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.wav")
    
    #Sử dụng bộ lọc Haar  để nhận diện khuôn mặt
    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    #Sử dụng bộ lọc Haar để nhận diện mắt trái
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    #Sử dụng bộ lọc Haar để nhận diện mắt phải
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


    #Tạo 1 list bao gồm 2 nhãn đóng và mở mắt
    lbl=['Close','Open']

    #Load model đã train được file .h5
    model = load_model('model_opencloseeyes_best.h5')
    #Hàm trả về đường dẫn thư mục hiện tại đang làm việc 
    path = os.getcwd()
    #Tạo font chữ 
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    #Khởi tạo bộ đếm và điểm đánh giá 
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            #Đọc từng khung hình từ video  
            height,width = frame.shape[:2] 
            #Chuyển đổi ảnh sang hệ gray 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            #Dectect ra khuôn mặt   
            faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
            #Detect ra mắt phải và trái, giá trị trả về gồm có tọa độ mắt kèm theo chiều dài và chiều rộng của hình chữ nhật bao quanh mắt
            left_eye = leye.detectMultiScale(gray)
            right_eye =  reye.detectMultiScale(gray)
            
            cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
            
            for (x,y,w,h) in faces:
                #Vẽ ra hình chữ nhật bao quanh khuôn mặt 
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            for (x,y,w,h) in left_eye:
                #Cắt ra hình ảnh mắt sau khi biết tọa độ 
                l_eye = frame[y:y+h,x:x+w]
                #Tăng bộ đếm lên 1 
                count=count+1
                #Các bước sử lý dữ liệu trước khi đưa vào mô hình để dự đoán 
                l_eye = cv2.resize(l_eye,(84,84))
                l_eye = np.array(l_eye).reshape(1, 84, 84, 3)
                l_eye = l_eye/255.0
                #Dự đoán mô hình, giá trị trả về gồm 1 mảng dữ liệu kích thước 1x2, giá trị gồm 0 và 1, bên trái là đóng, bên phải là mở. 1 = True, 0 = False
                lpred = (model.predict(l_eye) > 0.5).astype("int32")
                #Gán nhãn 
                if(lpred[0][0] == 1):
                    lbl='Closed'
                if(lpred[0][0] == 0):
                    lbl='Open'
                #Sau khi detect được thì thoát khỏi vòng lặp 
                break
                #Tương tự 
                
            for (x,y,w,h) in right_eye:
                r_eye = frame[y:y+h,x:x+w]
                count=count+1
                r_eye = cv2.resize(r_eye,(84,84))
                r_eye = np.array(r_eye).reshape(1, 84, 84, 3)
                r_eye = r_eye/255.0
                rpred = (model.predict(r_eye) > 0.5).astype("int32")
                if(rpred[0][0] == 1):
                    lbl='Closed'
                if(rpred[0][0] == 0):
                    lbl='Open'
                break
            
            #Kiểm tra nếu 1 mắt cùng nhắm thì vào hàm if
            if lpred[0][0] == 1 and rpred[0][0] == 1:
                #Điểm đánh giá cộng thêm 1 
                score=score+1
                cv2.putText(frame,"Closed",(10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
            else: 
                #Ngược lại điểm đánh giá trừ đi 1 
                score = score - 1
                cv2.putText(frame,"Open",(10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
            #Nếu Điểm đánh giá âm thì gán bằng không
            if(score<0):
                score = 0
            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
            #Nếu điểm đánh giá lớn hơn 15 thì đưa ra cả báo 
            if(score>15):
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pass
                cv2.imwrite(os.path.join(path,'image.jpg'), frame)
                if(thicc<16):
                    thicc = thicc - 2
                else: 
                    thicc = thicc + 2
                    if (thicc<2):
                        thicc = 2
                cv2.rectangle(frame, (0,0), (width, height), (0,255,255), thicc)
            
            success, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)
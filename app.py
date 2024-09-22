from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
app = Flask(__name__)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
json_file = open('emotion_model.json', 'r')
loaded_model = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model)
emotion_model.load_weights('emotion_model.weights.h5')
cap = cv2.VideoCapture(0)
face_detector=cv2.CascadeClassifier('haarcascade_frontalface.xml')
def gen_frames():
    while True:
        ret,frame=cap.read()
        # frame = cv2.resize(frame, (1280, 620))
        if not ret:
            break
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        for (x,y,w,h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_image= np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction=emotion_model.predict(cropped_image)
            max_index=int(np.argmax(emotion_prediction))
            cv2.putText(frame,emotion_dict[max_index],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)




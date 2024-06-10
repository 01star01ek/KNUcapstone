# onnx 이용 버전

import re
import cv2
from flask import Flask, render_template, Response
import numpy as np
import threading
import sys
import getopt

# Flask 애플리케이션 생성
app = Flask(__name__)

frame_lock = threading.Lock()
output_frame = None

# OpenCV 이용 ONNX 모델 로드
model = cv2.dnn.readNetFromONNX('bestonver.onnx')

# 웹캠 캡처 객체 생성
def create_capture(source = 0, fallback = 'synth:'):
    source = str(source).strip()

    # Win32: handle drive letter ('c:', ...)
    source = re.sub(r'(^|=)([a-zA-Z]):([/\\a-zA-Z0-9])', r'\1?disk\2?\3', source)
    chunks = source.split(':')
    chunks = [re.sub(r'\?disk([a-zA-Z])\?', r'\1:', s) for s in chunks]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )
    
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def capture_frames():
    import sys
    import getopt
    
    global output_frame, frame_lock
    
    args, sources = getopt.getopt(sys.argv[1:], '', 'shotdir=')
    args = dict(args)
    shotdir = args.get('--shotdir', '.')
    if len(sources) == 0:
        sources = [ 0 ]
    
    caps = list(map(create_capture, sources))
    
    while True:
        # 비디오 프레임 캡처
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                break
            
            with frame_lock:
                output_frame = frame.copy()

def detect_objects(frame):
    # JPEG 바이트 스트림을 numpy 배열로 변환
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    # 이미지가 numpy 배열인지 확인 및 변환
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    # 이미지가 uint8 형식인지 확인하고 변환
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # 이미지 전처리
    frame = cv2.resize(frame, (416, 416))
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, size=(416, 416), swapRB=True, crop=False)
    
    # 네트워크 입력 설정
    model.setInput(blob)
    
    # 추론 수행
    outputs = model.forward()
    
    # 결과 처리
    h, w = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{class_ids[i]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def gen_frames():
    global output_frame, frame_lock
    
    # 프레임 캡처 스레드 시작
    threading.Thread(target=capture_frames, daemon=True).start()
    
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            
            # 프레임 스트리밍
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
            
            # 객체 감지
            detect_objects(frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/main')
def main() :
   return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    # 비디오 스트리밍 경로
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':  
   # Flask 애플리케이션 실행
   app.run('0.0.0.0',port=5000,debug=True)

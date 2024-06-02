import re
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
from flask import Flask, render_template, Response
import numpy as np
import threading
import sys
import getopt

# Flask 애플리케이션 생성
app = Flask(__name__)

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestyolo.pt')

frame_lock = threading.Lock()
output_frame = None

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

# 이미지 변환
def preprocess_image(img):
	image = Image.fromarray(img)
	image = image.convert('RGB')
	transform = transforms.Compose([transforms.Resize((416,416)),])
	image = transform(image)
	return image

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
                
            # YOLOv5 이용 객체 감지
            frame = preprocess_image(frame)
            results = model(frame)
            # 결과 프레임에 박스 그리기
            frame = results.render()[0]
            
            with frame_lock:
                output_frame = frame.copy()

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


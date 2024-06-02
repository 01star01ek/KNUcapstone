import re

import cv2
from flask import Flask, render_template, Response
import numpy as np

# Flask 애플리케이션 생성
app = Flask(__name__)

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
    
# YOLO 모델 로드 및 설정
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = [] # 클래스 목록 지정
with open("coco.names", "r") as f:
	classes = f.read().strip().split("\n")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
# YOLO 객체 감지 실행
def detect_objects(frame):
	blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	# 감지된 객체를 프레임에 표시
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				# 객체가 감지되면 프레임에 표시
				center_x = int(detection[0] * frame.shape[1])
				center_y = int(detection[1] * frame.shape[0])
				w = int(detection[2] * frame.shape[1])
				h = int(detection[3] * frame.shape[0])
				cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
				cv2.putText(frame, classes[class_id], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def gen_frames():
    import sys
    import getopt

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
            
            # 객체 감지
            detect_objects(frame)
            
            # 프레임 스트리밍
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # 루트 경로에 접속했을 때 웹 페이지 렌더링
    return "<h1>OpenCV Video Streaming with Flask</h1><img src=\"/video_feed\">"

@app.route('/video_feed')
def video_feed():
    # 비디오 스트리밍 경로
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Flask 애플리케이션 실행
    app.run(host='0.0.0.0', port=5000, debug=True)


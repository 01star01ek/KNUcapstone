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
            
            # 프레임 스트리밍
            ret, buffer = cv2.imencode('.jpg', frame)
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

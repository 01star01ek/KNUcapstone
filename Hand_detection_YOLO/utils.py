import os, sys
import cv2
import time
import numpy as np
import torch
from tqdm import tqdm
import re
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_avi():
    actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
                'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
    label_dict = dict()

    for i in range(len(actions)):
        label_dict[actions[i]] = i

    videoFolderPath = "../SignLanguagedataset/output_video"
    videoTestList = os.listdir(videoFolderPath)
    
    testTargetList =[]

    for videoPath in videoTestList:
        actionVideoPath = f'{videoFolderPath}/{videoPath}'
        actionVideoList = os.listdir(actionVideoPath)
        for actionVideo in actionVideoList:
            fullVideoPath = f'{actionVideoPath}/{actionVideo}'
            testTargetList.append(fullVideoPath)

    # 모든 영상을 한번에 전처리하면 메모리 초과 문제 발생
    i = 12
    batch_size = 30
    
    testTargetList = testTargetList[i * batch_size : (i+1) * batch_size]
    print(f"batch_size = {batch_size}, data = {len(testTargetList)}")

    #testTargetList = sorted(testTargetList, key=lambda x:x[x.find("/", 9)+1], reverse=True)

    return testTargetList, label_dict

def make_directory(output_dir, data_folder):
    folder = f"{output_dir}/{data_folder}"
    image_folder = f"{output_dir}/{data_folder}/images"
    labels_folder = f"{output_dir}/{data_folder}/labels"

    if not os.path.exists(folder):
        os.makedirs(image_folder)
        os.makedirs(labels_folder)
    else:
        # 폴더가 이미 존재하는 경우
        # image_files > label_files : YOLO가 confidence(0.5)이상 확률로 예측하는 경우만 label를 사용하기 때문에
        # 손이 안 보이거나 noise가 많은 image는 label이 없는 파일이 존재함. ex) test/ㄹ_10_43.txt
        image_files = os.listdir(image_folder)
        label_files = os.listdir(labels_folder)
        print(f"{data_folder} = image({len(image_files)}), label({len(label_files)})")

def save_frames_and_labels(video_paths, label_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = []
    for path in tqdm(video_paths, desc="Processing videos"):
        
        pattern = re.compile('|'.join(label_dict.keys()))
        sign = list(set(pattern.findall(path)))[0]
        sign_id = label_dict[sign]

        cap = cv2.VideoCapture(path)

        # 동영상 파일을 열 수 없는 경우
        if not cap.isOpened():
            logging.error(f"Failed to open video: {path}")
            cap.release() 
            continue  

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_filename = f"{os.path.basename(path).split('.')[0]}_{frame_count}.jpg"
            label_filename = f"{os.path.basename(path).split('.')[0]}_{frame_count}.txt"
            
            data.append((frame, sign_id, img_filename, label_filename))
            frame_count += 1
            
        cap.release()

    return data

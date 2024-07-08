import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from yolo import YOLO
from utils import load_avi, make_directory, save_frames_and_labels
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

video_paths, label_dict = load_avi()
print(f"testTargetList : 영상개수 : {len(video_paths)}") 
print("label_dict : ", label_dict)

output_dir = '../dataset'
make_directory(output_dir, data_folder="train")
make_directory(output_dir, data_folder="valid")
make_directory(output_dir, data_folder="test")

data = save_frames_and_labels(video_paths, label_dict, output_dir)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

print("train : valid : test : ", len(train_data), len(valid_data), len(test_data))

print("\nstarting webcam...")
cv2.namedWindow("preview")

sets = ['train', 'valid', 'test']

for dataset, set_name in zip([train_data, valid_data, test_data], sets):
    for frame, sign_id, img_filename, label_filename in tqdm(dataset, desc = f"{set_name} process"):
        image_filename = f"{output_dir}/{set_name}/images/{img_filename}"
        label_filename = f"{output_dir}/{set_name}/labels/{label_filename}"
        
        # 영상을 fps(=30)단위로 쪼갠 이미지 저장
        cv2.imwrite(image_filename, frame)
        
        # 이미지 내 손가락 검출
        # confidence(0.5)보다 낮을 경우, 즉 Hand Detection이 불확실한 경우 예측을 하지 않음
        width, height, inference_time, results = yolo.inference(frame)

        results.sort(key=lambda x: x[2])

        # how many hands should be shown
        hand_count = len(results)
        if args.hands != -1:
            hand_count = int(args.hands)

        # 실제로 label저장시는 정규화된 경계를 저장(크기에 따라 다른 모양이 생기므로)
        # display hands
        for detection in results[:hand_count]:
            id, name, confidence, x, y, w, h = detection

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height

            with open(label_filename, 'w') as f:
                f.write(f"{sign_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

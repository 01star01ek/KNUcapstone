# YOLO-Hand-Detection

- 참고 github : https://github.com/cansik/yolo-hand-detection/tree/master
- 이 프로젝트는 YOLO 기반으로 자모음을 분류하기 위한 레이블 설정을 진행합니다. 자모음 카테고리는 사전에 분류할 수 있지만, 손의 bounding box 설정은 수작업으로 하기에는 시간이 많이 소요됩니다. 따라서 해당 GitHub의 hand_detection_model을 활용하여 예측한 모델을 불러와 사용하며, 실제로 몇몇 사진 및 영상을 확인한 결과 큰 오류는 없었습니다.
  
### Inferencing
모델은 416x416 크기의 이미지에서 학습되었습니다. 속도를 높이기 위해 더 작은 크기의 이미지로 추론할 수도 있습니다. 256x256 크기의 이미지를 사용하면 CPU에서 성능과 정확도의 좋은 균형을 이룰 수 있습니다.


### Demo
데모를 실행하려면, 가중치를 모델 폴더에 다운로드하십시오 (또는 셸 스크립트를 실행하십시오).
```bash
# mac / linux
cd models && sh ./download-models.sh

# windows
cd models && powershell .\download-models.ps1

```

그런 다음 YOLOv3를 사용하는 demo_webcam.py를 시작하십시오:

```bash
# with python 3
python demo_webcam.py
```

또는 YOLOv3-tiny를 사용하는 demo_webcam.py를 시작하십시오:
Or this one to run a webcam detrector with YOLOv3 tiny:

```bash
# with python 3
python demo_webcam.py -n tiny
```

또는 YOLOv3-Tiny-PRN를 사용하는 demo_webcam.py를 시작하십시오:

```bash
# with python 3
python demo_webcam.py -n prn
```

또는 YOLOv4-Tiny를 사용하는 demo_webcam.py를 시작하십시오:

```bash
# with python 3
python demo_webcam.py -n v4-tiny
```

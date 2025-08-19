# 《A Review of Research on Steel Defect Detection Based on YOLO Series Algorithms》Related documents

## 1. Project Overview
In this paper, we conduct in-depth research on relevant technical papers, take steel defect detection as the background, comprehensively sort out the development of YOLO series algorithms, and elaborate on various aspects of improvement in each version. Through a comprehensive benchmark analysis on the GC10-DET dataset containing various steel surface defects, the model performance is evaluated by indicators such as accuracy, time and model complexity, and the entropy weight method is used to objectively evaluate each model.

## 2. Environment configuration
Install the dependency library pip install -r requirements.txt (this project environment is configured according to the yolov12 project environment)
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolo python=3.11
conda activate yolo
pip install -r requirements.txt
pip install -e .
```


## 3. Model Training
We will yolov3, yolov5, yolov8, yolov9, yolov10, yolov11, yolov12 and yolov13 all use the YOLO module of ultralytics for training, while yolov6 and yolov7 do not support YOLO module. You need to download and configure the environment in https://github.com/meituan/YOLOv6 and https://github.com/WongKinYiu/yolov7 respectively. Localize some codes according to the official documentation prompts, and then set hyperparameters to start training


| Model |
| ---- |
| yolov3-tinyu |
| yolov3u |
| yolov5nu |
| yolov5su |
| yolov5mu |
| yolov5lu |
| yolov5xu |
| yolov6n |
| yolov6s |
| yolov6m |
| yolov6l |
| yolov7 |
| yolov7x |
| yolov8n |
| yolov8s |
| yolov8m |
| yolov8l |
| yolov8x |
| yolov9t |
| yolov9s |
| yolov9m |
| yolov9c |
| yolov9e |
| yolov10n |
| yolov10s |
| yolov10m |
| yolov10b |
| yolov10l |
| yolov10x |
| yolo11n |
| yolo11s |
| yolo11m |
| yolo11l |
| yolo11x |
| yolov12n |
| yolov12s |
| yolov12m |
| yolov12l |
| yolov12x |
| yolov13n |
| yolov13s |
| yolov13l |
| yolov13x |

Since our research did not change the architecture or other parameters of the standard model, we used all the coco pretrained weights for auxiliary training during training, and saved the trained model

COCO pretrained weights is obtained through Baidu Netdisk:

Link: https://pan.baidu.com/s/1n6Go0_cWw3okxy-oYIfN_w?pwd=t8dg 

Extraction Code: t8dg 

Trained model is obtained through Baidu Netdisk:

Link: https://pan.baidu.com/s/10_ROgPeEZdrNIEuaKGSULw?pwd=qeq8

Extraction Code: qeq8
Set the following hyperparameters (all unmentioned hyperparameters are by default) for training:
```bash
epochs = 400
batch_size = 16
batch_size = 8 # yolov12x and yolov13x
img_size = 640
```

The GC10-DET dataset can be obtained through Baidu Netdisk：

Link: https://pan.baidu.com/s/1VG9pjvp0z4qtAjfiridiKg?pwd=riyz Extraction Code: riyz


## 4. Model evaluation and visualization

Evaluation metrics selected mAP, Precision, Recall, Preprocess Time, Inference Time, Postprocess Time, Total Time, GFLOPs, and Model Size metrics
According to the evaluation index, we use the entropy weight to evaluate the model comprehensively, and get the final model performance and model weight, and save it in the evaluation_results .xlsx file


## 5. Acknowledgments
we are indebted to the open-source projects that have provided invaluable support:
- YOLOv3: https://github.com/ultralytics/yolov3
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv6: https://github.com/meituan/YOLOv6
- YOLOv7: https://github.com/WongKinYiu/yolov7
- YOLOv8: https://github.com/ultralytics/ultralytics
- YOLOv9: https://github.com/WongKinYiu/yolov9
- YOLOv10: https://github.com/THU-MIG/yolov10
- YOLOv11: https://github.com/ultralytics/ultralytics
- YOLOv12: https://github.com/sunsmarterjie/yolov12
- YOLOv13: https://github.com/iMoonLab/yolov13
- OpenCV: https://opencv.org/
    

import time
import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from thop import profile
import os
import json
import csv


def get_metrics(model_path, data_yaml):
    try:
        model = YOLO(model_path)
        val_results = model.val(data=data_yaml)
        metrics = {
            'mAP50': val_results.results_dict['metrics/mAP50(B)'],
            'mAP50_95': val_results.results_dict['metrics/mAP50-95(B)'],
            'Precision': val_results.results_dict['metrics/precision(B)'],
            'Recall': val_results.results_dict['metrics/recall(B)'],
            'Preprocess Time': val_results.speed['preprocess'],
            'Inference Time': val_results.speed['inference'],
            'Postprocess Time': val_results.speed['postprocess']
        }
        metrics['Total Time'] = metrics['Preprocess Time'] + metrics['Inference Time'] + metrics['Postprocess Time']
        return metrics
    except Exception as e:
        print(f"Error getting metrics for {model_path}: {e}")
        return None


def get_gflops(model_path, device, img_size=640):
    try:
        model = YOLO(model_path)
        model.model = model.model.to(device)
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        flops, _ = profile(model.model, inputs=(dummy_input,))
        gflops = flops / 1e9
        return gflops
    except Exception as e:
        print(f"Error getting GFLOPs for {model_path}: {e}")
        return None


def get_model_size(model_path):
    try:
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # 转换为 MB
        return model_size
    except Exception as e:
        print(f"Error getting model size for {model_path}: {e}")
        return None

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the list of models to be evaluated
    model_paths = [
        'detect/yolov3u.pt/weights/best.pt',
        'detect/yolov3-tinyu.pt/weights/best.pt',
        'detect/yolov5nu.pt/weights/best.pt',
        'detect/yolov5su.pt/weights/best.pt',
        'detect/yolov5mu.pt/weights/best.pt',
        'detect/yolov5lu.pt/weights/best.pt',
        'detect/yolov5xu.pt/weights/best.pt',
        'detect/yolov8n.pt/weights/best.pt',
        'detect/yolov8s.pt/weights/best.pt',
        'detect/yolov8m.pt/weights/best.pt',
        'detect/yolov8l.pt/weights/best.pt',
        'detect/yolov8x.pt/weights/best.pt',
        'detect/yolov9t.pt/weights/best.pt',
        'detect/yolov9s.pt/weights/best.pt',
        'detect/yolov9m.pt/weights/best.pt',
        'detect/yolov9c.pt/weights/best.pt',
        'detect/yolov9e.pt/weights/best.pt',
        'detect/yolov10n.pt/weights/best.pt',
        'detect/yolov10s.pt/weights/best.pt',
        'detect/yolov10m.pt/weights/best.pt',
        'detect/yolov10b.pt/weights/best.pt',
        'detect/yolov10l.pt/weights/best.pt',
        'detect/yolov10x.pt/weights/best.pt',
        'detect/yolo11n.pt/weights/best.pt',
        'detect/yolo11s.pt/weights/best.pt',
        'detect/yolo11m.pt/weights/best.pt',
        'detect/yolo11l.pt/weights/best.pt',
        'detect/yolo11x.pt/weights/best.pt',
        'detect/yolov12n.pt/weights/best.pt',
        'detect/yolov12s.pt/weights/best.pt',
        'detect/yolov12m.pt/weights/best.pt',
        'detect/yolov12l.pt/weights/best.pt' ,
        'detect/yolov12x.pt/weights/best.pt' ,
        'detect/yolov13n.pt/weights/best.pt',
        'detect/yolov13s.pt/weights/best.pt',
        'detect/yolov13l.pt/weights/best.pt',
        'detect/yolov13x.pt/weights/best.pt' 
    ]
    data_yaml = '/root/GC-DET/data.yaml'


    # Define the header of the CSV file
    fieldnames = [
        'Model', 'mAP50', 'mAP50_95', 'Precision', 'Recall',
        'Preprocess Time', 'Inference Time', 'Postprocess Time',
        'Total Time', 'GFLOPs', 'Model Size'
    ]

    # Open the CSV file and write the header
    with open('evaluation_results.csv', mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Traverse each model for evaluation
        for model_path in model_paths:
            print(f"Evaluating model: {model_path}")

            # Acquire mAP, Precision, Recall, Time Metrics and FPS
            metrics = get_metrics(model_path, data_yaml)
            if metrics is None:
                continue

            # Get GFLOPs
            gflops = get_gflops(model_path, device)
            if gflops is None:
                continue

            # Get model size
            model_size = get_model_size(model_path)
            if model_size is None:
                continue

            # Merge all metrics
            all_metrics = {
                'Model': model_path,
                **metrics,
                'GFLOPs': gflops,
                'Model Size': model_size
            }

            # Write evaluation results to CSV file
            writer.writerow(all_metrics)

    print("Evaluation completed. Results saved to evaluation_results.csv")

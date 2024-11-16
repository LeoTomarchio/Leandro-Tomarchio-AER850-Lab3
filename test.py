from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import torch

# Load your trained model
model = YOLO('runs/detect/pcb_detector/weights/best.pt')

# List of images to predict
images = [
    "data/evaluation/ardmega.jpg",
    "data/evaluation/arduno.jpg",
    "data/evaluation/rasppi.jpg"
]

# Run prediction on each image
for img in images:
    # Predict with confidence threshold of 0.25
    results = model(img, conf=0.25)
    
    # Save results
    for r in results:
        # Plot detection boxes on image
        im_array = r.plot()
        # Save the annotated image
        cv2.imwrite(f'prediction_{os.path.basename(img)}', im_array)
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import torch

def process_motherboard_image(image_path):
    """
    Process motherboard image using advanced OpenCV masking techniques
    Args:
        image_path (str): Path to the input image
    Returns:
        masked_image: Processed image with applied mask
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded")
    
    # Convert to grayscale for threshold and edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Edge detection using Canny
    edges = cv2.Canny(gray, 100, 200)
    
    # Corner detection using Harris
    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 1000  # Adjust this value based on your image
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create mask from filtered contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), -1)
    
    # Apply mask to original image
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    
    return {
        'original': img,
        'threshold': thresh,
        'edges': edges,
        'corners': harris_corners,
        'mask': mask,
        'masked_image': masked_image
    }

def save_processing_steps(results, output_dir='output'):
    """
    Save all processing steps as images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each processing step
    cv2.imwrite(f"{output_dir}/1_original.jpg", results['original'])
    cv2.imwrite(f"{output_dir}/2_threshold.jpg", results['threshold'])
    cv2.imwrite(f"{output_dir}/3_edges.jpg", results['edges'])
    cv2.imwrite(f"{output_dir}/4_corners.jpg", results['corners'] * 255)  # Scale corners for visibility
    cv2.imwrite(f"{output_dir}/5_mask.jpg", results['mask'])
    cv2.imwrite(f"{output_dir}/6_masked_image.jpg", results['masked_image'])

def detect_pcb_components(image_path, model_path):
    """
    Detect and classify PCB components using YOLOv8
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained YOLO model
    Returns:
        results: YOLOv8 detection results
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)
    return results

def train_pcb_model(data_yaml_path, epochs=150, batch=8, imgsz=640):
    """
    Train YOLOv8 model for PCB component detection
    """
    # Get absolute path to data.yaml
    data_yaml_abs_path = os.path.abspath(data_yaml_path)
    
    # Initialize the model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml_abs_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name='pcb_detector',
        patience=20,
        save=True,
        device='0',
        verbose=True,
        workers=4,
        exist_ok=True,
        cache=False,
        amp=False
    )
    
    return model

def main():
    # Single CUDA check at start
    if torch.cuda.is_available():
        print(f"Training will use GPU: {torch.cuda.get_device_name()}")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("No GPU available, training will use CPU")
    
    # Paths
    motherboard_image_path = "motherboard_image.JPEG"
    data_yaml_path = "data.yaml"
    
    # Train PCB detection model
    try:
        print("Starting model training...")
        model = train_pcb_model(
            data_yaml_path=data_yaml_path,
            epochs=150,
            batch=8,
            imgsz=640
        )
        print("Training completed successfully!")
        
        # Test the trained model on motherboard image
        results = model(motherboard_image_path)
        
        # Save detection results
        for r in results:
            im_array = r.plot()
            cv2.imwrite('pcb_detection_results.jpg', im_array)
            
    except Exception as e:
        print(f"Error in model training or inference: {e}")

if __name__ == "__main__":
    main()

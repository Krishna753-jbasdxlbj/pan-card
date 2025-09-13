import os
import shutil
import yaml
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
import logging
from typing import List, Dict, Tuple

class PANCardOCRSystem:
    def __init__(self, project_path: str = r"C:\pan card"):
        self.project_path = Path(project_path)
        self.setup_logging()
        self.class_names = {
            0: 'Pan',
            1: 'PanNo', 
            2: 'Name',
            3: 'Father_Name',
            4: 'DOB',
            5: 'Signature'
        }
        self.model = None
        self.ocr_reader = easyocr.Reader(['en'])
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'pan_ocr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_dataset_config(self):
        """Create YAML configuration file for YOLO training"""
        config = {
            'path': str(self.project_path / 'dataset'),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 6,  # number of classes
            'names': list(self.class_names.values())
        }
        
        config_path = self.project_path / 'configs' / 'pan_card_config.yaml'
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Dataset configuration created at {config_path}")
        return config_path

    def organize_dataset(self, source_images_path: str, source_labels_path: str):
        """
        Organize images and labels into train/val/test splits
        Args:
            source_images_path: Path to folder containing all images
            source_labels_path: Path to folder containing all label files
        """
        source_img_dir = Path(source_images_path)
        source_label_dir = Path(source_labels_path)
        
        # Get all image files
        image_files = list(source_img_dir.glob('*.jpg')) + list(source_img_dir.glob('*.png')) + list(source_img_dir.glob('*.jpeg'))
        image_files = sorted(image_files)
        
        if len(image_files) != 71:
            self.logger.warning(f"Expected 71 images, found {len(image_files)}")
        
        # Split ratios: 70% train, 20% val, 10% test
        train_split = int(0.7 * len(image_files))  # ~50 images
        val_split = int(0.9 * len(image_files))    # ~15 images for val, rest for test
        
        splits = {
            'train': image_files[:train_split],
            'val': image_files[train_split:val_split],
            'test': image_files[val_split:]
        }
        
        for split_name, files in splits.items():
            img_dir = self.project_path / 'dataset' / split_name / 'images'
            label_dir = self.project_path / 'dataset' / split_name / 'labels'
            
            for img_file in files:
                # Copy image
                shutil.copy2(img_file, img_dir / img_file.name)
                
                # Copy corresponding label file
                label_file = source_label_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    shutil.copy2(label_file, label_dir / label_file.name)
                else:
                    self.logger.warning(f"Label file not found: {label_file}")
            
            self.logger.info(f"Organized {len(files)} files for {split_name} split")

    def train_model(self, epochs: int = 50, batch_size: int = 16, img_size: int = 640):
        """Train YOLOv8 model on PAN card dataset"""
        try:
            # Load YOLOv8n model
            self.model = YOLO('yolov8n.pt')
            
            # Create config file
            config_path = self.create_dataset_config()
            
            # Training parameters
            training_args = {
                'data': str(config_path),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'project': str(self.project_path / 'results'),
                'name': 'pan_card_detection',
                'save_period': 10,  # Save checkpoint every 10 epochs
                'patience': 15,     # Early stopping patience
                'optimizer': 'AdamW',
                'lr0': 0.01,       # Initial learning rate
                'weight_decay': 0.0005,
                'augment': True,   # Enable augmentation
                'mosaic': 0.5,     # Mosaic augmentation probability
                'copy_paste': 0.1, # Copy-paste augmentation probability
                'device': 'cuda' if self.check_gpu() else 'cpu'
            }
            
            self.logger.info(f"Starting training with {epochs} epochs...")
            results = self.model.train(**training_args)
            
            # Save the best model
            best_model_path = self.project_path / 'weights' / 'best_pan_card_model.pt'
            shutil.copy2(results.save_dir / 'weights' / 'best.pt', best_model_path)
            
            self.logger.info(f"Training completed! Best model saved at {best_model_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load_trained_model(self, model_path: str = None):
        """Load trained model for inference"""
        if model_path is None:
            model_path = self.project_path / 'weights' / 'best_pan_card_model.pt'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = YOLO(str(model_path))
        self.logger.info(f"Model loaded from {model_path}")

    def detect_and_extract(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect regions and extract text from PAN card
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
        Returns:
            Dictionary containing extracted information
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run YOLO detection
        results = self.model(image, conf=confidence_threshold)
        
        extracted_data = {
            'image_path': image_path,
            'detections': [],
            'extracted_text': {}
        }
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, 'Unknown')
                    
                    # Crop region
                    cropped_region = image[y1:y2, x1:x2]
                    
                    # Extract text using EasyOCR
                    ocr_results = self.ocr_reader.readtext(cropped_region)
                    extracted_text = ' '.join([text[1] for text in ocr_results if text[2] > 0.5])
                    
                    detection_info = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'text': extracted_text
                    }
                    
                    extracted_data['detections'].append(detection_info)
                    extracted_data['extracted_text'][class_name] = extracted_text
        
        return extracted_data

    def process_batch(self, input_folder: str, output_folder: str = None):
        """Process multiple images in batch"""
        if output_folder is None:
            output_folder = self.project_path / 'output'
        
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        input_path = Path(input_folder)
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
        
        results = []
        
        for img_file in image_files:
            try:
                self.logger.info(f"Processing {img_file.name}...")
                result = self.detect_and_extract(str(img_file))
                results.append(result)
                
                # Save individual result
                with open(output_path / f"{img_file.stem}_result.json", 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Create annotated image
                self.create_annotated_image(str(img_file), result, output_path / f"{img_file.stem}_annotated.jpg")
                
            except Exception as e:
                self.logger.error(f"Error processing {img_file.name}: {str(e)}")
        
        # Save batch results
        with open(output_path / 'batch_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Batch processing completed. Results saved in {output_path}")
        return results

    def create_annotated_image(self, image_path: str, detection_result: Dict, output_path: str):
        """Create annotated image with bounding boxes and extracted text"""
        image = cv2.imread(image_path)
        
        for detection in detection_result['detections']:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            text = detection['text']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add extracted text (if short enough)
            if len(text) < 20:
                cv2.putText(image, text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        cv2.imwrite(output_path, image)

    def evaluate_model(self, test_data_path: str = None):
        """Evaluate model performance"""
        if test_data_path is None:
            test_data_path = self.dataset_path / 'images' / 'test'
        
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Run evaluation
        results = self.model.val(data=str(self.create_dataset_config()))
        
        self.logger.info("Model evaluation completed")
        return results

def main():
    """Main execution function"""
    # Initialize the system
    pan_ocr = PANCardOCRSystem(r"C:\pan card")
    
    print("PAN Card OCR Detection System")
    print("=" * 50)
    
    while True:
        print("\nSelect an option:")
        print("1. Organize dataset (split images into train/val/test)")
        print("2. Train YOLOv8 model")
        print("3. Load trained model and process images")
        print("4. Evaluate model")
        print("5. Process batch of images")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        try:
            if choice == '1':
                source_images = input("Enter path to source images folder: ").strip()
                source_labels = input("Enter path to source labels folder: ").strip()
                pan_ocr.organize_dataset(source_images, source_labels)
                
            elif choice == '2':
                epochs = int(input("Enter number of epochs (default 50): ") or "50")
                batch_size = int(input("Enter batch size (default 16): ") or "16")
                pan_ocr.train_model(epochs=epochs, batch_size=batch_size)
                
            elif choice == '3':
                model_path = input("Enter model path (or press Enter for default): ").strip()
                if model_path:
                    pan_ocr.load_trained_model(model_path)
                else:
                    pan_ocr.load_trained_model()
                
                image_path = input("Enter path to image to process: ").strip()
                result = pan_ocr.detect_and_extract(image_path)
                
                print("\nExtracted Information:")
                for class_name, text in result['extracted_text'].items():
                    print(f"{class_name}: {text}")
                
            elif choice == '4':
                pan_ocr.load_trained_model()
                results = pan_ocr.evaluate_model()
                print("Evaluation completed. Check logs for details.")
                
            elif choice == '5':
                input_folder = input("Enter input folder path: ").strip()
                output_folder = input("Enter output folder path (or press Enter for default): ").strip()
                pan_ocr.load_trained_model()
                results = pan_ocr.process_batch(input_folder, output_folder or None)
                print(f"Processed {len(results)} images")
                
            elif choice == '6':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
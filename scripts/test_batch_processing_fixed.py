# test_batch_processing_fixed.py
import sys
import os
from pathlib import Path
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

sys.path.append(str(Path(__file__).parent / 'scripts'))

try:
    from pan_card_ocr import PANCardOCRSystem
except ImportError as e:
    print(f"Error importing: {e}")
    exit(1)

def test_batch_processing():
    print("PAN Card OCR - Batch Processing Test (Fixed)")
    print("=" * 50)
    
    # Initialize system
    try:
        pan_ocr = PANCardOCRSystem()
        pan_ocr.load_trained_model()
        print("System and model loaded")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    # Get input folder
    input_folder = input("Enter folder path with PAN card images: ").strip().strip('"')
    
    if not os.path.exists(input_folder):
        print(f"Folder not found: {input_folder}")
        return
    
    try:
        print(f"Processing all images in: {input_folder}")
        print("This may take several minutes...")
        
        # Process batch (modified to use our custom encoder)
        results = []
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            print(f"Processing {image_file}...")
            
            try:
                result = pan_ocr.detect_and_extract(image_path, confidence_threshold=0.5)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results.append({'image_path': image_path, 'error': str(e)})
        
        # Save results with custom encoder
        os.makedirs("test_results", exist_ok=True)
        with open('test_results/batch_summary.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"Batch processing completed!")
        print(f"Processed {len(results)} images")
        print(f"Results saved in: test_results folder")
        
        # Summary statistics
        successful_extractions = 0
        total_detections = 0
        
        for result in results:
            if 'extracted_text' in result and result['extracted_text']:
                successful_extractions += 1
            if 'detections' in result:
                total_detections += len(result['detections'])
        
        print(f"Summary:")
        print(f"   Successful extractions: {successful_extractions}/{len(results)}")
        print(f"   Total objects detected: {total_detections}")
        print(f"   Average detections per image: {total_detections/len(results):.1f}")
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_processing()
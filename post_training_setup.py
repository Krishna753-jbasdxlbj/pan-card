# 📋 PAN Card OCR System - Post-Training Guide
# This script contains all the steps needed after training completion

import os
import sys
import json
import subprocess
from pathlib import Path
import shutil

def main():
    print("🚀 PAN Card OCR System - Post-Training Setup")
    print("=" * 50)
    
    # Set the base directory
    base_dir = r"C:\pan card"
    os.chdir(base_dir)
    
    # STEP 1: Verify Training Success
    print("\n1. Verifying Training Success...")
    verify_training_success(base_dir)
    
    # STEP 2: Create Testing Environment
    print("\n2. Creating Testing Environment...")
    create_test_environment(base_dir)
    
    # STEP 3: Create Test Scripts
    print("\n3. Creating Test Scripts...")
    create_test_scripts(base_dir)
    
    # STEP 4: Instructions for Testing
    print("\n4. Testing Instructions:")
    print_testing_instructions()
    
    # STEP 5: Create Production Script
    print("\n5. Creating Production Script...")
    create_production_script(base_dir)
    
    # STEP 6: Performance Analysis
    print("\n6. Performance Analysis Setup...")
    create_performance_scripts(base_dir)
    
    # STEP 7: Usage Workflow
    print("\n7. Usage Workflow Created")
    
    # STEP 8: Troubleshooting Guide
    print("\n8. Troubleshooting Guide Available")
    
    # STEP 9: Final Checklist
    print("\n9. Final Checklist:")
    print_final_checklist()
    
    print("\n🎉 Setup Complete! Your PAN Card OCR system is ready for testing and deployment.")

def verify_training_success(base_dir):
    """Step 1: Verify training completed successfully"""
    print("Checking training completion...")
    
    # Check if model files exist
    models_dir = os.path.join(base_dir, "models")
    runs_dir = os.path.join(base_dir, "runs")
    
    model_files = []
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
    
    run_dirs = []
    if os.path.exists(runs_dir):
        run_dirs = os.listdir(runs_dir)
    
    print(f"Model files found: {len(model_files)}")
    print(f"Training runs found: {len(run_dirs)}")
    
    # Check for best model
    best_model = os.path.join(models_dir, "best_pan_card_model.pt")
    if os.path.exists(best_model):
        print("✅ Best model found!")
    else:
        print("❌ Best model not found. Checking runs directory...")
        # Look for model files in runs directory
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                if file.endswith('.pt') and 'best' in file.lower():
                    best_model = os.path.join(root, file)
                    print(f"Found model in runs: {best_model}")
                    break

def create_test_environment(base_dir):
    """Step 2: Create testing environment"""
    test_dirs = [
        os.path.join(base_dir, "test_images"),
        os.path.join(base_dir, "test_results"),
        os.path.join(base_dir, "sample_outputs")
    ]
    
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Try to copy sample images if available
    source_dir = os.path.join(base_dir, "datasets", "test", "images")
    dest_dir = os.path.join(base_dir, "test_images")
    
    if os.path.exists(source_dir):
        try:
            # Copy a few images for testing
            image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for i, file in enumerate(image_files[:3]):  # Copy first 3 images
                src_path = os.path.join(source_dir, file)
                dst_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied sample image: {file}")
        except Exception as e:
            print(f"Note: Could not copy sample images: {e}")
    else:
        print("Note: No sample images found in datasets/test/images/")

def create_test_scripts(base_dir):
    """Step 3: Create test scripts"""
    
    # Single image test script
    single_test_script = """
import sys
import os
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

try:
    from pan_card_ocr import PANCardOCRSystem
except ImportError as e:
    print(f"Error importing: {e}")
    print("Make sure pan_card_ocr.py exists in scripts folder")
    exit(1)

def test_single_image():
    print("🔍 PAN Card OCR - Single Image Test")
    print("=" * 40)
    
    # Initialize system
    try:
        pan_ocr = PANCardOCRSystem()
        print("✅ System initialized")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Load trained model
    try:
        pan_ocr.load_trained_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Make sure training completed and model exists in models folder")
        return
    
    # Get image path
    while True:
        image_path = input("\\n📁 Enter path to PAN card image (or 'quit' to exit): ").strip().strip('\"')
        
        if image_path.lower() == 'quit':
            break
            
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        try:
            print(f"\\n🔄 Processing: {os.path.basename(image_path)}")
            print("Please wait...")
            
            # Process image
            result = pan_ocr.detect_and_extract(image_path, confidence_threshold=0.5)
            
            print("\\n" + "="*50)
            print("🎉 EXTRACTION RESULTS")
            print("="*50)
            
            if result['extracted_text']:
                for field, text in result['extracted_text'].items():
                    print(f"📋 {field:15}: {text}")
            else:
                print("❌ No text extracted. Try lowering confidence threshold.")
            
            print(f"\\n📊 Detection Summary:")
            print(f"   Total detections: {len(result['detections'])}")
            
            for detection in result['detections']:
                print(f"   - {detection['class']}: {detection['confidence']:.3f} confidence")
            
            # Save results
            output_file = f"test_results/{os.path.splitext(os.path.basename(image_path))[0]}_result.txt"
            with open(output_file, 'w') as f:
                f.write("PAN Card OCR Results\\n")
                f.write("="*30 + "\\n")
                for field, text in result['extracted_text'].items():
                    f.write(f"{field}: {text}\\n")
            
            print(f"💾 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_single_image()
"""
    
    with open(os.path.join(base_dir, "test_single_image.py"), "w") as f:
        f.write(single_test_script)
    print("✅ Created single image test script")
    
    # Batch test script
    batch_test_script = """
import sys
import os
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent / 'scripts'))

try:
    from pan_card_ocr import PANCardOCRSystem
except ImportError as e:
    print(f"Error importing: {e}")
    exit(1)

def test_batch_processing():
    print("📚 PAN Card OCR - Batch Processing Test")
    print("=" * 45)
    
    # Initialize system
    try:
        pan_ocr = PANCardOCRSystem()
        pan_ocr.load_trained_model()
        print("✅ System and model loaded")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Get input folder
    input_folder = input("📁 Enter folder path with PAN card images: ").strip().strip('\"')
    
    if not os.path.exists(input_folder):
        print(f"❌ Folder not found: {input_folder}")
        return
    
    try:
        print(f"\\n🔄 Processing all images in: {input_folder}")
        print("This may take several minutes...")
        
        # Process batch
        results = pan_ocr.process_batch(input_folder, "test_results")
        
        print(f"\\n🎉 Batch processing completed!")
        print(f"📊 Processed {len(results)} images")
        print(f"💾 Results saved in: test_results folder")
        
        # Summary statistics
        successful_extractions = 0
        total_detections = 0
        
        for result in results:
            if result['extracted_text']:
                successful_extractions += 1
            total_detections += len(result['detections'])
        
        print(f"\\n📈 Summary:")
        print(f"   Successful extractions: {successful_extractions}/{len(results)}")
        print(f"   Total objects detected: {total_detections}")
        print(f"   Average detections per image: {total_detections/len(results):.1f}")
        
        # Save summary
        summary = {
            'total_images': len(results),
            'successful_extractions': successful_extractions,
            'total_detections': total_detections,
            'results': results
        }
        
        with open('test_results/batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved to: test_results/batch_summary.json")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_processing()
"""
    
    with open(os.path.join(base_dir, "test_batch_processing.py"), "w") as f:
        f.write(batch_test_script)
    print("✅ Created batch processing test script")

def print_testing_instructions():
    """Step 4: Print testing instructions"""
    instructions = """
Testing Instructions:
-------------------
1. Run single image test: python test_single_image.py
   - Enter path to a PAN card image when prompted
   - Wait for processing
   - Review extracted information
   - Check if all fields are detected correctly

2. Evaluate Results Quality:
   Good Results Should Show:
   ✅ All 6 classes detected (Pan, PanNo, Name, Father_Name, DOB, Signature)
   ✅ Confidence scores > 0.5
   ✅ Text extraction working for each field
   ✅ Accurate bounding box placement

3. If Results Are Poor:
   - Try with lower confidence threshold (modify confidence_threshold from 0.5 to 0.3)
   - Check if images are similar to training data
   - Verify label quality in training data

4. Test Batch Processing: python test_batch_processing.py
   - Enter folder path with multiple PAN card images
   - Review batch results in test_results folder
"""
    print(instructions)

def create_production_script(base_dir):
    """Step 5: Create production script"""
    production_script = """
import sys
import os
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent / 'scripts'))
from pan_card_ocr import PANCardOCRSystem

class PANCardProcessor:
    def __init__(self):
        self.pan_ocr = PANCardOCRSystem()
        self.pan_ocr.load_trained_model()
        
    def process_folder(self, input_folder, output_folder=None):
        if not output_folder:
            output_folder = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"🔄 Processing folder: {input_folder}")
        print(f"📁 Output folder: {output_folder}")
        
        results = self.pan_ocr.process_batch(input_folder, output_folder)
        
        # Create Excel/CSV summary
        import pandas as pd
        
        summary_data = []
        for result in results:
            row = {'filename': Path(result['image_path']).name}
            row.update(result['extracted_text'])
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{output_folder}/extraction_summary.csv", index=False)
        df.to_excel(f"{output_folder}/extraction_summary.xlsx", index=False)
        
        print(f"✅ Processing complete!")
        print(f"📊 Summary saved as Excel and CSV")
        
        return results

def main():
    print("🏭 PAN Card OCR - Production Processor")
    print("=" * 40)
    
    processor = PANCardProcessor()
    
    while True:
        print("\\nOptions:")
        print("1. Process single image")
        print("2. Process folder of images") 
        print("3. Exit")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip().strip('\"')
            result = processor.pan_ocr.detect_and_extract(image_path)
            
            print("\\nExtracted Information:")
            for field, text in result['extracted_text'].items():
                print(f"{field}: {text}")
                
        elif choice == '2':
            folder_path = input("Enter folder path: ").strip().strip('\"')
            processor.process_folder(folder_path)
            
        elif choice == '3':
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(base_dir, "production_processor.py"), "w") as f:
        f.write(production_script)
    print("✅ Created production processor script")

def create_performance_scripts(base_dir):
    """Step 6: Create performance analysis scripts"""
    performance_script = """
import sys
sys.path.append('scripts')
from pan_card_ocr import PANCardOCRSystem

def evaluate_performance():
    print("📊 Model Performance Evaluation")
    print("=" * 35)
    
    pan_ocr = PANCardOCRSystem()
    pan_ocr.load_trained_model()
    
    # Evaluate on test set
    try:
        results = pan_ocr.evaluate_model()
        print("✅ Evaluation completed!")
        print("Check runs folder for detailed metrics")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    evaluate_performance()
"""
    
    with open(os.path.join(base_dir, "evaluate_performance.py"), "w") as f:
        f.write(performance_script)
    print("✅ Created performance evaluation script")

def print_final_checklist():
    """Step 9: Print final checklist"""
    checklist = """
Final Checklist:
---------------
After training completion, ensure you can:
✅ Load the trained model successfully
✅ Process a single PAN card image  
✅ Extract all 6 fields (Pan, PanNo, Name, Father_Name, DOB, Signature)
✅ Process multiple images in batch
✅ Generate annotated images with bounding boxes
✅ Export results to JSON/CSV/Excel
✅ Achieve reasonable accuracy (>70% detection rate)

Quick Start Commands:
-------------------
# Test single image
python test_single_image.py

# Test batch processing  
python test_batch_processing.py

# Run production processor
python production_processor.py

# Evaluate model performance
python evaluate_performance.py

Troubleshooting:
--------------
Issue: Model not found
- Check if model exists in models/ folder
- Look for model files in runs/ directory

Issue: Low accuracy
- Lower confidence threshold (0.3 instead of 0.5)
- Check if images are similar to training data
- Verify label quality in training data

Issue: No text extracted
- Check detection first (bounding boxes)
- Verify EasyOCR is working
- Try with higher resolution images

Performance Optimization:
-----------------------
For faster processing, install GPU PyTorch if you have NVIDIA GPU:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""
    print(checklist)

if __name__ == "__main__":
    main()
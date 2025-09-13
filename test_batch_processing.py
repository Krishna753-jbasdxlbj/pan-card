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
        print(f"\n🔄 Processing all images in: {input_folder}")
        print("This may take several minutes...")
        
        # Process batch
        results = pan_ocr.process_batch(input_folder, "test_results")
        
        print(f"\n🎉 Batch processing completed!")
        print(f"📊 Processed {len(results)} images")
        print(f"💾 Results saved in: test_results folder")
        
        # Summary statistics
        successful_extractions = 0
        total_detections = 0
        
        for result in results:
            if result['extracted_text']:
                successful_extractions += 1
            total_detections += len(result['detections'])
        
        print(f"\n📈 Summary:")
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

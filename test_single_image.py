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
        image_path = input("\n📁 Enter path to PAN card image (or 'quit' to exit): ").strip().strip('\"')
        
        if image_path.lower() == 'quit':
            break
            
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        try:
            print(f"\n🔄 Processing: {os.path.basename(image_path)}")
            print("Please wait...")
            
            # Process image
            result = pan_ocr.detect_and_extract(image_path, confidence_threshold=0.5)
            
            print("\n" + "="*50)
            print("🎉 EXTRACTION RESULTS")
            print("="*50)
            
            if result['extracted_text']:
                for field, text in result['extracted_text'].items():
                    print(f"📋 {field:15}: {text}")
            else:
                print("❌ No text extracted. Try lowering confidence threshold.")
            
            print(f"\n📊 Detection Summary:")
            print(f"   Total detections: {len(result['detections'])}")
            
            for detection in result['detections']:
                print(f"   - {detection['class']}: {detection['confidence']:.3f} confidence")
            
            # Save results
            output_file = f"test_results/{os.path.splitext(os.path.basename(image_path))[0]}_result.txt"
            with open(output_file, 'w') as f:
                f.write("PAN Card OCR Results\n")
                f.write("="*30 + "\n")
                for field, text in result['extracted_text'].items():
                    f.write(f"{field}: {text}\n")
            
            print(f"💾 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_single_image()

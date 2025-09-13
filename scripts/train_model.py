import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pan_card_ocr import PANCardOCRSystem

def main():
    print("PAN Card Model Training")
    print("=" * 30)
    
    # Initialize system
    pan_ocr = PANCardOCRSystem()
    
    # Training parameters
    epochs = int(input("Enter number of epochs (default 50): ") or "50")
    batch_size = int(input("Enter batch size (default 16): ") or "16")
    
    print(f"\nStarting training with {epochs} epochs and batch size {batch_size}")
    print("This may take a while depending on your hardware...")
    
    try:
        # Train the model
        results = pan_ocr.train_model(epochs=epochs, batch_size=batch_size)
        print("\nTraining completed successfully!")
        print(f"Best model saved in: C:\\pan card\\models\\")
        print(f"Training results saved in: C:\\pan card\\runs\\")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
from ultralytics import YOLO
import yaml

# Create correct config
config = {
    'path': r'C:\pan card\datasets',
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images', 
    'nc': 6,
    'names': ['Pan', 'PanNo', 'Name', 'Father_Name', 'DOB', 'Signature']
}

# Save config
import os
os.makedirs('configs', exist_ok=True)
with open('configs/pan_card_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Config created with correct path!')

# Load model and train
model = YOLO('yolov8n.pt')
results = model.train(
    data='configs/pan_card_config.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    project='runs',
    name='pan_card_detection'
)
print('Training completed!')

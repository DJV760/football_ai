from ultralytics import YOLO

# load model
model = YOLO('yolov8x')

# run model predict
model.predict('input_videos/08fd33_4.mp4', save=True)
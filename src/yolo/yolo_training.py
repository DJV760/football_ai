from ultralytics import YOLO
import torch

def yolo_train():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using: {device}')

    model = YOLO('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\yolo\\yolo_weights\\yolov8x_fine_tune_rgb.pt').to(device)

    model.train(
        data='C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\jersey_number_detection_dataset_5\\data.yaml',   # Path to your dataset YAML file
        epochs=50,         
        imgsz=224,          
        batch=16,           
        workers=1,          
        name='yolo_digits', 
        optimizer='Adam',   
        patience=5,         
        val=True,
        device='cuda'
    )

    metrics = model.val()

    model.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\yolo\\yolo_weights\\yolov8x_fine_tune_grayscale.pt')
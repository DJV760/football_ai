from ultralytics import YOLO
import torch

def yolo_player_detection_train():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using: {device}')

    model = YOLO('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\yolo\\yolo_weights\\yolov8x.pt').to(device)

    model.train(
        data='C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\yolo\\player_detection\\player_detection_dataset_2\\data.yaml', 
        epochs=100,         
        imgsz=512,          
        batch=16,          
        workers=2,          
        name='yolo_player_detection',
        optimizer='Adam',   
        patience=7,        
        val=True,
        device='cuda'
    )

    metrics = model.val()

    model.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\yolo\\yolo_weights\\yolov8x_player_detection_3.pt')
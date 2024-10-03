from ultralytics import YOLO
import cv2

def draw_custom_annotations(image, results):

    '''
    This function draws custom annotations (bounding boxes and appropriate text) over the frame of a video

    Parameters:
        image: A raw frame from a video
        results:

    Returns:
        image: Processed frame, with annotation drawn over it
    '''

    for result in results:
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        labels = result.boxes.cls

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()
            score = scores[i].item()
            label = int(labels[i].item())

            color_map = {
                0: ['referee', (0, 0, 0)],
                1: ['ball', (255, 255, 255)],
                2: ['goalkeeper', (255, 0, 0)],
                3: ['player', (0, 255, 0)]
            }

            center_x = int((box[0] + box[2]) / 2)
            center_y = int(box[3])
            axis_length_x = int((box[2] - box[0]) / 2)
            axis_length_y = int((box[3] - box[1]) / 8)

            cv2.ellipse(image, (center_x, center_y), (axis_length_x, axis_length_y), 0, -45, 235, color_map[label][1], 3)
            
            text = f'{color_map[label][0]}, ({score:.2f})'
            font_scale = 0.5
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y

            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_map[label][1], thickness)

    return image




# Load model
model = YOLO('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\yolo\\yolo_weights\\yolov5x_player_detection_closeup.pt')

# Load video file
video_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\input_videos\\input_video.mp4'
output_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\output_videos\\output_video_players_yolo5xft_closeup_4k.mp4' 

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO inference on the current frame
    results = model(frame)
    
    # Visualize the results by drawing bounding boxes
    annotated_frame = draw_custom_annotations(frame, results)

    # Write the frame with annotations to the output video
    out.write(annotated_frame)

    # Optionally display the frame (press 'q' to exit)
    cv2.imshow('YOLO Inference', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
 
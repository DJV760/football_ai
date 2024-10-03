from ultralytics import YOLO
import cv2

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
    annotated_frame = results[0].plot()  # Draw bounding boxes

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

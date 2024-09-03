import cv2

# initialize minimum probability to eliminate weak predictions
p_min = 0.5
thres = 0.

# 'VideoCapture' object and reading vicv2.mean(image, mask=mask)deo from a file
video = cv2.VideoCapture('input_videos\\08fd33_4.mp4')
writer = None
h, w = None, None

# Create labels into list
with open('coco.names') as f:
    labels = [line.strip() for line in f]

# load network 
network = cv2.dnn.readNet('darknet/cfg/yolov3.weights', 'darknet/cfg/yolov3.cfg')

# Getting only output layer names that we need from YOLO
ln = network.getLayerNames()
ln = [ln[i - 1] for i in network.getUnconnectedOutLayers()]

# Defining loop for catching frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    if h is None or w is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter('output_videos/output.mp4', fourcc, fps, (w, h))

    # frame preprocessing for deep learning
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities.
    network.setInput(blob)
    output_from_network = network.forward(ln)
    #process bounding boxes in output_from_network

    # Write the processed frame to the output video
    writer.write(frame)

    # Optional: Display the video while processing
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and reader
writer.release()
video.release()
cv2.destroyAllWindows()
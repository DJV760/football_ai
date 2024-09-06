import cv2
from RealESRGAN import RealESRGAN
from PIL import Image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the image
image_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\jersey_number_detection_dataset_3\\train\\images\\cropped_image_000001-_00_jpg.rf.2acf74d6729c5d6aad2e935381ae1aed.jpg'
image = Image.open(image_path).convert('RGB')

model = RealESRGAN(device, scale=4)
model.load_weights('number_classification\RealESRGAN_x4plus.pth', download=True)

sr_image = model.predict(image)
sr_image.show()
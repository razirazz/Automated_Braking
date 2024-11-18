import numpy as np
import json
import os
import cv2
from ultralytics import YOLO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


# def plot_image_with_details(image_name, detected_objects, min_distance, brake):
#     # Load the image
#     image = Image.open(image_name).convert('RGB')
#     draw = ImageDraw.Draw(image)
#
#     # Draw detected objects
#     for obj in detected_objects:
#         bbox = obj['bbox']  # assuming bbox is in the format [x1, y1, x2, y2]
#         label = obj['label']
#         color = 'red' if brake else 'green'
#
#         # Draw bounding box
#         draw.rectangle(bbox, outline=color, width=3)
#
#         # Draw label
#         draw.text((bbox[0], bbox[1]), label, fill=color)
#
#     # Plot image with details
#     plt.figure(figsize=(10, 10))
#     plt.imshow(np.array(image))
#     plt.title(f"Min Distance: {min_distance:.2f}, Brake: {'Yes' if brake else 'No'}")
#     plt.axis('on')
#     plt.show()

# Install and load YOLOv8 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8s', pretrained=True)
model = YOLO('yolov8n.pt')
# model.train(data="coco8.yaml", epochs=10)

# Function to detect objects using YOLOv8
def detect_objects(image):
    results = model(image)
    detected_objects = []
    # print(f'results = {results}')
    for boxes in results:
        # print('\nboxes = ', boxes.boxes)
        # print(f'data.data.shape = {boxes.boxes.data.shape}')
        for data in boxes.boxes:
            # print('\ncls = ', data.cls.item())
            # print('\nconf = ', data.conf)
            # print('\ndata = ', data.data)
            x1, y1, x2, y2, conf, cls = data.data[0]
            # print(f'x1, y1, x2, y2, conf, cls = {x1, y1, x2, y2, conf, cls}')
            detected_objects.append({
                # print(f'boxes.names[int(cls) = {boxes.names[int(cls)]}'),
                # print(f'[x1.item(), y1.item(), x2.item(), y2.item()] = {[x1.item(), y1.item(), x2.item(), y2.item()]}')
                'label': boxes.names[int(cls)],
                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                'confidence': conf.item()
            })
    return detected_objects

# Function to calculate distance from the bottom-middle of the image
def calculate_distance(bbox, image_shape):
    bottom_middle = (image_shape[1] / 2, image_shape[0] - 70)
    object_center = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)
    # print(f'object_center = {object_center}')
    distance = np.sqrt((object_center[0] - bottom_middle[0]) ** 2 + (object_center[1] - bottom_middle[1]) ** 2)
    return distance

# Directory containing images
# image_dir = 'D:/project/automatedBraking/imageDataset/test'
image_dir = 'D:/project/automatedBraking/imageDataset/train'
# image_dir = 'D:/project/automatedBraking/dataSet100k/train'

# Create a list to store annotations
annotations = []

# Loop through images in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Detect objects in the image
    detected_objects = detect_objects(image)

    # Calculate distances and determine if braking is needed
    min_distance = float('inf')
    brake = False

    for obj in detected_objects:
        distance = calculate_distance(obj['bbox'], image.shape)
        obj['distance'] = distance
        if distance < min_distance:
            min_distance = distance
        # if 40 <= distance < 250:
        #     brake = True

    if 40 <= min_distance < 250:
        brake = True

    # Create annotation for the image
    annotation = {
        'image_name': image_name,
        'objects': detected_objects,
        'min_distance': min_distance,
        'brake': brake
    }

    # plot_image_with_details(image_path, detected_objects, min_distance, brake)

    annotations.append(annotation)

# Save annotations to a JSON file
with open('annotations_10000.json', 'w') as f:
    json.dump(annotations, f, indent=4)
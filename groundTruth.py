import json
import os

# Load the annotations
with open('annotations_10000.json') as f:
    annotations = json.load(f)

# Create a list to store ground truth data
ground_truth = []

# Loop through annotations
for annotation in annotations:
    image_name = annotation['image_name']
    objects = annotation['objects']

    # Find the true minimum distance and brake data
    min_distance = float('inf')
    brake = False

    for obj in objects:
        distance = obj['distance']
        if distance < min_distance:
            min_distance = distance
        if 40 <= distance < 250:
            brake = True

    # Create ground truth data for the image
    gt = {
        'image_name': image_name,
        'min_distance': min_distance,
        'brake': brake
    }

    ground_truth.append(gt)

# Save ground truth data to a JSON file
with open('ground_truth_10000.json', 'w') as f:
    json.dump(ground_truth, f, indent=4)

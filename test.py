import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import os
import glob
import json
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def save_label(image_path, prediction, output_path):
    label_data = {
        "image": os.path.basename(image_path),
        "brake": prediction
    }
    with open(output_path, 'w') as f:
        json.dump(label_data, f)

def calculate_distances(image_path, predictions, score_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    bottom_middle = np.array([width / 2, height])

    distances = []

    if predictions is None:
        return None  # No obstacles detected above the score threshold

    print(f'predictions = {predictions.shape}')

    for i in range(len(predictions)):
        score = predictions.item()
        print(f'score = {score}')
        if score >= score_threshold:
            box = predictions[i].cpu().numpy()
            if len(box) >= 4:  # Ensure box has enough dimensions
                obstacle_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                distance = np.linalg.norm(obstacle_center - bottom_middle)
                distances.append(distance)

    if distances:
        min_distance = min(distances)
    else:
        min_distance = None  # No valid obstacles detected above the score threshold

    return min_distance


# Load the saved model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load('brake_model_10000.pth'))
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

def detect_obstacles(image_path):
    image = Image.open(image_path).convert("RGB")
    # print(f'transform(image) = {transform(image)}')
    image_tensor = transform(image).unsqueeze(0)
    # print(f'image_tensor = {image_tensor}')

    with torch.no_grad():
        output = model(image_tensor)
        # print(f'output = {output}')
        _, predicted = torch.max(output, 1)
        # print(f'predicted.item() = {predicted.item()}')
    return predicted.item()

def process_image_dataset(image_folder, output_folder, score_threshold=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
    # print(len(image_paths))
    for image_path in image_paths:
        predictions = detect_obstacles(image_path)
        # min_distance = calculate_distances(image_path, predictions, score_threshold)
        output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '.json'))
        save_label(image_path, predictions, output_path)

def evaluate_performance(ground_truth_file, prediction_folder):
    # ground_truth_files = glob.glob(os.path.join(ground_truth_folder, '*.json'))
    predictions_files = glob.glob(os.path.join(prediction_folder, '*.json'))

    ground_truths = []
    predictions = []

    # print(f'ground_truth_files = {ground_truth_file}')
    for data in ground_truth_file:
        # print('gt_data[brake]', data['brake'])
        if data['brake']:
            ground_truths.append(1)
        else:
            ground_truths.append(0)


    for pred_file in predictions_files:
        with open(pred_file) as f:
            pred_data = json.load(f)
        predictions.append(pred_data['brake'])

    # print(f'ground_truths, predictions = {ground_truths, predictions}')

    mse = mean_squared_error(ground_truths, predictions)
    accuracy = accuracy_score(ground_truths, predictions)
    # conf_matrix = confusion_matrix(ground_truths, predictions)
    class_report = classification_report(ground_truths, predictions, target_names=['No Brake', 'Brake'])
    # tn, fp, fn, tp = conf_matrix.ravel()
    # specificity, also called Precision = tp / (tp + fp)
    # sensitivity, also called Recall = tp / (tp + fn)

    print(f'Mean Squared Error: {mse}')
    print(f'Accuracy: {accuracy}')
    print(class_report)

    ConfusionMatrixDisplay.from_predictions(ground_truths, predictions)
    plt.show()

# Example usage
test_image_folder = 'D:/project/automatedBraking/imageDataset/test'
output_folder = 'D:/project/automatedBraking/imageDataset/test/labels'
process_image_dataset(test_image_folder, output_folder)

# ground_truth_file = 'D:/project/automatedBraking/ground_truth_test.json'
with open('D:/project/automatedBraking/ground_truth_test_10000.json') as f:
    ground_truth_file = json.load(f)
prediction_folder = output_folder
evaluate_performance(ground_truth_file, prediction_folder)


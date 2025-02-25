import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import random
import time
import sys
import pyvww
import sys
#sys.path.append("/home/giannis/mcunet")  # Ensure Python finds MCUNet
from mcunet.model_zoo import net_id_list, build_model, download_tflite

model_type = sys.argv[1]
model_path = sys.argv[2]
grid_x= int(sys.argv[3])
grid_y= int(sys.argv[4])
overlap= float(sys.argv[5])
device= sys.argv[6]


def split_image_tensor(image_tensor, grid_size, overlap):
    _, img_height, img_width = image_tensor.shape  # (C, H, W)
    num_rows, num_cols = grid_size

    patch_width = img_width // num_cols
    patch_height = img_height // num_rows

    overlap_w = int(patch_width * overlap)
    overlap_h = int(patch_height * overlap)

    patches = []

    for row in range(num_rows):
        for col in range(num_cols):
            left = max(col * patch_width - overlap_w, 0)
            upper = max(row * patch_height - overlap_h, 0)
            right = min(left + patch_width + overlap_w, img_width)
            lower = min(upper + patch_height + overlap_h, img_height)

            patch = image_tensor[:, upper:lower, left:right]  # Slice tensor directly
            patches.append(patch)

    return patches


def inference(model_path, model, device, new_validation_set, grid_x, grid_y, overlap):
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    correct_pred = 0
    total_correct_patches = 0
    start_time = time.time()

    with torch.no_grad():
        for image, label in new_validation_set:
            image, label = image.unsqueeze(0).to(device), torch.tensor(label).to(device) #tensor(label)
            
            # Full image classification
            output = model(image)
            _, predicted = torch.max(output, 1)
            if predicted.item() == label.item():
                correct_pred += 1

    end_time = time.time()

    start_patch_time = time.time()
    with torch.no_grad():
        for image, label in new_validation_set:
            image, label = image.unsqueeze(0).to(device), torch.tensor(label).to(device)
            
            patches = split_image_tensor(image.squeeze(0), (grid_x, grid_y), overlap)
            patches_tensor = torch.stack([patch.unsqueeze(0) for patch in patches]).to(device)  # Shape: (batch_size, channels, height, width)


            if len(patches_tensor.shape) == 5:
                patches_tensor = patches_tensor.squeeze(1)

            outputs = model(patches_tensor)

            person_count = 0

            for output in outputs:
                _, pred = torch.max(output, 0)
                if pred.item() == 1:
                    person_count += 1
            
            correct_prediction = (person_count > 0) == (label.item() == 1)
            if correct_prediction:
                total_correct_patches += 1
            #print({person_count},{label})
    end_patch_time = time.time()

    # Compute Accuracy
    accuracy = (correct_pred / 150) * 100
    accuracy_4 = (total_correct_patches / 150) * 100
    fps = 150 / (end_time - start_time)
    fps_patch = 150 / (end_patch_time - start_patch_time)

    print(f"\n✅ Device: {device}, Accuracy on 150 images: {accuracy:.2f}%, FPS: {fps:.2f}, Grid:({grid_x,grid_y}) Patch Accuracy: {accuracy_4:.2f}%, FPS: {fps_patch:.2f}")

    return accuracy, accuracy_4, fps, fps_patch

if model_type == 'mcunet':
    model, image_size, description = build_model(net_id="mcunet-vww2", pretrained=True)  # You can change net_id
    num_features = model.classifier.linear.in_features  # Get last FC layer input size
    model.classifier.linear = nn.Linear(num_features, 2)  # Change output to 2 classes
    transform = transforms.Compose([
        transforms.Resize((144, 144)), ##144, 144 for mcunet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if model_type == 'mobilenet':
        model = models.mobilenet_v3_small(weights=None)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 2)

    elif model_type == 'resnet':
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

    elif model_type == 'shuffle_0_5':
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    elif model_type == 'shuffle_1_0':
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    else:
        model = models.squeezenet1_1(weights=None)
        model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))


new_validation_set = pyvww.pytorch.VisualWakeWordsClassification(
    root="../new-vww-subset/",
    annFile="../new-vww-subset/instances_val_subset.json",
    transform=transform
)

inference(model_path, model, device, new_validation_set, grid_x, grid_y, overlap)

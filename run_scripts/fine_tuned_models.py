import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pyvww
import time
from mcunet.model_zoo import net_id_list, build_model, download_tflite


model_type = sys.argv[1]
model_path = sys.argv[2]
batch = int(sys.argv[3])
device = sys.argv[4]


def stats(model, device, validation_loader):
    
    val_correct=0
    start_time = time.time()

    for vinputs, vlabels in validation_loader:
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)

        voutputs = model(vinputs)

        _, vpreds = torch.max(voutputs, 1)
        val_correct += (vpreds == vlabels).sum().item()
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time:.2f} seconds')
    print(f'Quantized Model Accuracy: {val_correct/150}')
    print(f'Frames Per Second: {execution_time/150:.3f} sec')

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


model.load_state_dict(torch.load(model_path, weights_only=True))

validation_set = pyvww.pytorch.VisualWakeWordsClassification(
    root="../new-vww-subset/",
    annFile="../new-vww-subset/instances_val_subset.json",
    transform=transform
)

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch, shuffle=True)

model.to(device)
model.eval()
stats(model, device, validation_loader)


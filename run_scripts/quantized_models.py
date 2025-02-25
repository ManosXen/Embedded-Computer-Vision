import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pyvww
import time
import copy 

model_type = sys.argv[1]
model_path = sys.argv[2]
batch = int(sys.argv[3])
device = 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_set = pyvww.pytorch.VisualWakeWordsClassification(
    root="../new-vww-subset/",
    annFile="../new-vww-subset/instances_val_subset.json",
    transform=transform
)

observation_loader = pyvww.pytorch.VisualWakeWordsClassification(
    root="../observer_dataset/",
    annFile="../observer_dataset/truncated_annotations.json",
    transform=transform
)

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch, shuffle=True)
observation_loader = torch.utils.data.DataLoader(observation_loader, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)

def ptsq(model):

    device = torch.device('cuda')
    device_cpu = torch.device('cpu')

    backend = "fbgemm" #x86

    m = copy.deepcopy(model)

    quantized_model = nn.Sequential(torch.quantization.QuantStub(), 
                  m, 
                  torch.quantization.DeQuantStub())

    quantized_model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(quantized_model, inplace=True)

    for vinputs, _ in observation_loader:
        if vinputs.dim() == 3:  # If input is (C, H, W), add batch dimension
            vinputs = vinputs.unsqueeze(0)  # Becomes (1, C, H, W)
        vinputs = vinputs.to(device_cpu)  # Move to CPU
        quantized_model(vinputs)
    
    torch.quantization.convert(quantized_model, inplace=True)
    
    return quantized_model


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
    print(f'Frames Per Second: {150/execution_time:.3f} sec')

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


quantized = ptsq(model)
state_dict = torch.load(model_path)
quantized.load_state_dict(state_dict)


quantized.to(device)
quantized.eval()
stats(quantized, device, validation_loader)


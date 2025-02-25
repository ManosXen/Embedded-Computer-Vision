import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pyvww
import time
import copy 

batch = int(sys.argv[1])
device = 'cpu'

base_model_path="../models/full_models/"
quantized_model_path = "../models/quantized_models/static/"

all_models=[("shuffle_0_5", "shufflenet_v2_x0_5.pth", "shufflenet_v2_x0_5.pt"), ("shuffle_1_0", "shufflenet_v2_x1_0.pth", "shufflenet_v2_x1_0.pt"), ("squeeze", "squeezenet1_1.pth", "squeezenet1_1.pt")]

output_file = "subframes_parallel_cpu.txt"

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


def stats(model, device, validation_loader, num_trials=10):
    val_correct = 0

    start_time = time.time()
    
    for _ in range(num_trials):
        for vinputs, vlabels in validation_loader:
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            _, vpreds = torch.max(voutputs, 1)
            val_correct += (vpreds == vlabels).sum().item()
    end_time = time.time()
    
    execution_time = (end_time - start_time) / num_trials
    
    return execution_time

for model in all_models:
    model_type=model[0]

    if model_type == 'shuffle_0_5':
        nonq_model = models.shufflenet_v2_x0_5(weights=None)
        nonq_model.fc = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    elif model_type == 'shuffle_1_0':
        nonq_model = models.shufflenet_v2_x1_0(weights=None)
        nonq_model.fc = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    else:
        nonq_model = models.squeezenet1_1(weights=None)
        nonq_model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    
    quantized = ptsq(nonq_model)
    state_dict = torch.load(quantized_model_path + model[2])
    quantized.load_state_dict(state_dict)

    quantized.to(device)

    quantized.eval()

    state_dict = torch.load(base_model_path + model[1])
    nonq_model.load_state_dict(state_dict)

    nonq_model.to(device)
    nonq_model.eval()

    exec_time_nquatized = stats(nonq_model, device, validation_loader)
    exec_time_quantized = stats(quantized, device, validation_loader)

    print(f'Model {model_type}: {exec_time_nquatized/exec_time_quantized:.3f}')




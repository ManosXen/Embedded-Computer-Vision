import os
import sys

device=sys.argv[1]

# (x, y, overlap)
configs = [(2, 2, 0.4), (2, 4, 0.4), (4, 2, 0.4)]

base_model_path="../models/full_models/"
models=[("mobilenet", "mobilenet_v3_Small_freeze.pth"), ("mobilenet", "mobilenet_v3_small_nofreeze.pth"), ("resnet", "resnet18.pth"), ("shuffle_0_5", "shufflenet_v2_x0_5.pth"), ("shuffle_1_0", "shufflenet_v2_x1_0.pth"), ("squeeze", "squeezenet1_1.pth")]

base_command="../bin/python3 check_models.py "
output_file = "subframes_parallel_cpu.txt"

for model in models:
    for config in configs:
        model_path = base_model_path + model[1]
        command = base_command + model[0] + " " + model_path + " " + str(config[0]) + " " + str(config[1]) + " " + str(config[2]) + " " + device + " >> " + output_file
        #command = base_command + model[0] + " " + model_path + " " + str(config[0]) + " " + str(config[1]) + " " + str(config[2]) + " " + device
        print(command)
        os.system("echo " + command)
        os.system(command)
        os.system("echo ")
        print()

os.system(f"grep -E '^(../bin|✅)' {output_file} > filtered_{output_file}")

os.rename(f"filtered_{output_file}", output_file)
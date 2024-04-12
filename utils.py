
from tensorflow.config import list_physical_devices

def print_gpu_availability():
    gpus = list_physical_devices('GPU')
    if gpus:
        print("GPU is available and recognized by TensorFlow!")
        for gpu in gpus:
            print("Name:", gpu.name, "Type:", gpu.device_type)
    else:
        print("GPU not available or not recognized by TensorFlow.")

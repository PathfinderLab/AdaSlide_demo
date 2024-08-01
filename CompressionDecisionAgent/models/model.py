import timm
import torch

def define_model(model_name="resnet18", num_classes=2):
    return timm.create_model(model_name, num_classes=num_classes, pretrained=True)

def load_compression_agent(model_path, model_name="resnet18", num_classes=2):
    model = define_model(model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    
    return model
    
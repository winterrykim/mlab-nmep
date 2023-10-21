from .lenet import LeNet
from .resnet import ResNet18
from .alexnet import AlexNet

def build_model(config):
    "Model builder."

    model_type = config.MODEL.NAME

    if model_type == 'lenet':
        model = LeNet(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'resnet18':
        model = ResNet18(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'alexnet':
        model = AlexNet(num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'resnet34':
    #     model = ResNet34(num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'resnet50':
    #     model = ResNet50(num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'resnet101':
    #     model = ResNet101(num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'resnet152':
    #     model = ResNet152(num_classes=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    
    return model
 
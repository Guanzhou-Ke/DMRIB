from .resnet import *


__all__ = ['build_off_the_shelf_cnn', 'build_mlp', 'get_norm', 'get_act']

def build_off_the_shelf_cnn(name='resnet18', 
                            fc_identity=True, 
                            num_classes=1000, 
                            channels=3,
                            maxpooling=True):
    """
    Build off-the-shelf CNN networks, likes ResNet, VGG...
    """
    builders = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152
    }
    func = builders.get(name, None)
    if func is None:
        raise ValueError(f"{name} is not a available network name.")
    backbone = func(num_classes=num_classes, channels=channels, fc_identity=fc_identity, maxpooling=maxpooling)
    return backbone


def build_mlp(layers, activation='relu', norm='batch', first_norm=True):
    """Build multiple linear perceptron

    Args:
        layers (list): The list of input and output dimension.
        activation (str, optional): activation function. Defaults to 'relu'.
                                    ['none', 'relu', 'softmax', 'sigmoid']
        norm (str, optional): normalization. Defaults to 'batch'.
                              `none`: not set, `batch`: denotes BatchNorm1D;
                              `layer`: denotes LayerNorm.
        first_norm (bool, optional): put the normalization layer before 
                                      non-liner activation function if True.
    """
    net = []
    for idx in range(1, len(layers)):
        net.append(
            nn.Sequential(
                nn.Linear(layers[idx-1], layers[idx]),
                get_norm(norm, num_features=layers[idx], dim=1) if first_norm else get_act(activation),
                get_act(activation) if first_norm else get_norm(norm, num_features=layers[idx], dim=1),
            )
        )
    net = nn.Sequential(*net)
    return net


def get_act(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(dim=-1)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'leaky-relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Activation function name error: {name}")
    

def get_norm(name, num_features, dim=1):
    if name == None:
        return nn.Identity()
    elif name == 'batch':
        return nn.BatchNorm1d(num_features) if dim == 1 else nn.BatchNorm2d(num_features)
    elif name == 'layer':
        return nn.LayerNorm(num_features)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Normalization name erro: {name}")
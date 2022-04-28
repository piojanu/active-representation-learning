from functools import partial

from .convnet import ConvNet26x26Encoder, ConvNet48x48Encoder
from .resnet import ResNetEncoder

# Deep means more channels and output features are 128 dim.
DeepResNet9Encoder = partial(ResNetEncoder, layers=[2, 2], channels=[64, 128])

# Shallow means less channels and output features are 64 dim.
ShallowConvNet48x48Encoder = partial(ConvNet48x48Encoder, n_features=64)
ShallowResNet9Encoder = partial(ResNetEncoder, layers=[2, 2], channels=[32, 64])
ShallowResNet5Encoder = partial(ResNetEncoder, layers=[1, 1], channels=[32, 64])

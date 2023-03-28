# Model for unlearning domain
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
########################################################################################################################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature = nn.Sequential()      # Define the feature extractor
        self.feature.add_module('f_conv1_1', nn.Conv2d(1, 4, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_1_1', nn.ReLU(True))
        self.feature.add_module('f_bn1_1', nn.BatchNorm2d(4))
        self.feature.add_module('f_conv1_2', nn.Conv2d(4, 4, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_1_2', nn.ReLU(True))
        self.feature.add_module('f_bn1_2', nn.BatchNorm2d(4))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))

        self.feature.add_module('f_conv2_1', nn.Conv2d(4, 8, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_2_1', nn.ReLU(True))
        self.feature.add_module('f_bn2_1', nn.BatchNorm2d(8))
        self.feature.add_module('f_conv2_2', nn.Conv2d(8, 8, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_2_2', nn.ReLU(True))
        self.feature.add_module('f_bn2_2', nn.BatchNorm2d(8))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))

        self.embeddings = nn.Sequential()
        self.embeddings.add_module('r_fc1', nn.Linear(49*8, 32))

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(-1, 49*8)
        feature_embedding = self.embeddings(feature)
        return feature_embedding

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.regressor = nn.Sequential()
        self.regressor.add_module('r_dropout', nn.Dropout(p=0.25))
        self.regressor.add_module('r_relu1', nn.ReLU(False))
        self.regressor.add_module('r_fc2', nn.Linear(32, 16))
        self.regressor.add_module('r_dropout', nn.Dropout(p=0.25))
        self.regressor.add_module('r_relu2', nn.ReLU(False))
        self.regressor.add_module('r_pred', nn.Linear(16, 11))
        self.regressor.add_module('r_pred_softmax', nn.Softmax(dim=1))

    def forward(self, x):
        regression = self.regressor(x)
        return regression



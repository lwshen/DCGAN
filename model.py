import torch.nn as nn
import torchvision

class discriminator(nn.Module):
    def __init__(self, alpha=0.2, dropout_value=0.0):
        super(discriminator, self).__init__()
        self.layers = nn.Sequential(
            # nn.LeakyReLU(alpha),
            nn.Conv2d(1, 64, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(64, 128, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(128, 256, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(256, 512, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(512, 1024, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.AvgPool2d((4, 4), 1, 0),
        )
        self.fc = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class generater(nn.Module):
    def __init__(self, dropout_value=0.0):
        super(generater, self).__init__()
        # input = 4 * 4 * 1024
        # self.ic = input_channel
        # self.oc = output_channel
        self.feature_collect = feature_collect()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(512, 256, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(256, 128, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(128, 64, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(64, 1, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh(),
        )

        self.weight_init()

    def forward(self, x):
        x = self.feature_collect(x)
        x = self.layers(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class feature_collect(nn.Module):
    def __init__(self):
        super(feature_collect, self).__init__()
        # input = 1*128*128 output = 1024*4*4
        res = torchvision.models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.conv2 = nn.Conv2d(256, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        # print(res)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x



import torch
import torch.nn as nn

# def conv3x3(in_channels, out_channels, **kwargs):
#     '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

#     layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#                           nn.ReLU(),
#                           nn.MaxPool3d(2, 2))

#     return layer


def upsample(scale_factor=2, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True),
                          )

    return layer


def _conv3x3(in_channels, out_channels, **kwargs):
                          
    layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                          nn.ReLU())

    return layer

class SegNet(nn.Module):
    def __init__(self, n_classes, n_input_channels, scaling_factor, skip='cat'):
        super(SegNet, self).__init__()

        self._skip = skip
        # Encoding path
        self.pool = nn.MaxPool3d(2, 2)

        self.conv1 = _conv3x3(n_input_channels, 32*scaling_factor)

        self.conv2 = _conv3x3(32*scaling_factor, 64*scaling_factor)

        self.conv3 = _conv3x3(64*scaling_factor, 128*scaling_factor)

        self.lat = nn.Conv3d(128*scaling_factor, 256*scaling_factor, 
                             kernel_size=3, padding=1)
        self.relu_lat = nn.ReLU()

        # Decoding path
        self.upsample = upsample()

        self.conv4 = _conv3x3(256*scaling_factor + 128*scaling_factor, 128*scaling_factor)

        self.conv5 = _conv3x3(128*scaling_factor + 64*scaling_factor, 64*scaling_factor)

        self.conv6 = _conv3x3(64*scaling_factor + 32*scaling_factor, 32*scaling_factor)

        self.conv_out = nn.Conv3d(32*scaling_factor, n_classes, kernel_size=1)

        self.activation = nn.Softmax(dim=3)
    def forward(self, x):
        # Encoding path
        x1 = self.conv1(x)
        x1_ = self.pool(x1)

        x2 = self.conv2(x1_)
        x2_ = self.pool(x2)

        x3 = self.conv3(x2_)
        x3_ = self.pool(x3)

        lat = self.relu_lat(self.lat(x3_))

        # Decoding path
        up1 = self.upsample(lat)

        if self._skip == 'cat':
            cat1 = torch.cat([up1, x3], dim=1)
        elif self._skip == 'add':
            cat1 = torch.add([up1, x3], dim=1)
      
        x4 = self.conv4(cat1)

        up2 = self.upsample(x4)

        if self._skip == 'cat':
            cat2 = torch.cat([up2, x2], dim=1)
        elif self._skip == 'add':
            cat2 = torch.add([up2, x2], dim=1)
      
        x5 = self.conv5(cat2)

        up3 = self.upsample(x5)

        if self._skip == 'cat':
            cat3 = torch.cat([up3, x1], dim=1)
        elif self._skip == 'add':
            cat3 = torch.add([up3, x1], dim=1)


        x6 = self.conv6(cat3)

        x = self.conv_out(x6)

        return self.activation(x)

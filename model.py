import torch
import torch.nn as nn
import os
import cv2
import numpy
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
        self.model = self.buildBlock(dim, use_dropout)

    def buildBlock(self, dim, use_dropout, pm=64):
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=pm * 4, out_channels=pm * 4, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(pm * 4),
            nn.ReLU(True),

            nn.Dropout(p=0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=pm * 4, out_channels=pm * 4, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(pm * 4)
        )
        return self.model

    def forward(self, input):
        # Add skip connection
        out = input + self.model(input)
        return out


class Generator_Res(nn.Module):
    def __init__(self, inchannels=3, outchannels=3, pm=64, use_dropout=True):
        super(Generator_Res, self).__init__()
        self.model = nn.Sequential(

            # Downsampling
            nn.ReflectionPad2d(3),

            nn.Conv2d(in_channels=inchannels, out_channels=pm * 1, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(pm * 1),
            nn.ReLU(True),

            nn.Conv2d(in_channels=pm * 1, out_channels=pm * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(pm * 2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=pm * 2, out_channels=pm * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(pm * 4),
            nn.ReLU(True),

            # Resblocks # 1
            ResnetBlock(dim=64, use_dropout=use_dropout),
            # Resblocks # 2
            ResnetBlock(dim=64, use_dropout=use_dropout),
            # Resblocks # 3
            ResnetBlock(dim=64, use_dropout=use_dropout),
            # Resblocks # 4
            ResnetBlock(dim=64, use_dropout=use_dropout),
            # Resblocks # 5
            ResnetBlock(dim=64, use_dropout=use_dropout),
            # Resblocks # 6
            ResnetBlock(dim=64, use_dropout=use_dropout),

            # Upsampling
            nn.ConvTranspose2d(in_channels=pm * 4, out_channels=pm * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(pm * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=pm * 2, out_channels=pm * 1, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(pm * 1),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(pm, outchannels, 7),
            nn.Tanh()
        )


    def forward(self, x):
        return self.model(x)
class Discriminator_Patch(nn.Module):
    def __init__(self, inchannels = 3, outchannels=1, pm=64):
        super(Discriminator_Patch, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=pm *1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels = pm * 1, out_channels=pm * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(pm * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=pm * 2, out_channels=pm * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(pm * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=pm * 4, out_channels=pm * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(pm * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=pm * 8, out_channels=outchannels, kernel_size=4, stride=1, padding=1)
        )


    def forward(self, x):
        return self.model(x)

def saveimage(tensorlist, filepath, unique:bool=False):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if unique:
        x = 1
        for file in os.listdir(filepath):
            if os.path.isfile(os.path.join(os.getcwd(), filepath, file)):
                x += 1

    else:
        x = 1
    firstdir = os.getcwd()
    dr = os.path.join(os.getcwd(), filepath)
    for a in tensorlist:
        with torch.no_grad():
            numpy_image = a.cpu().permute(1, 2, 0).numpy()

            numpy_image = numpy_image * 255
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
            img = cv2.imshow("image", numpy_image)
            os.chdir(dr)
            cv2.imwrite("img"+str(x)+".png", numpy_image)

        x += 1
    os.chdir(firstdir)
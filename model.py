from typing import Any
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.init as init
from torch.nn import functional as F
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
}


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2)
        self.conv1_relu = nn.ReLU()
        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        self.conv10_deconv = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)

        self._fc1 = nn.Linear(725355, 256)
        self._fc2 = nn.Linear(725355, 256)
        self._fc3 = nn.Linear(725355, 256)


    def extract_features(self, input):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """

        #     convolution     #
        conv1 = self.conv1(input)
        relu = self.conv1_relu(conv1)
        fire2 = self.fire2(nn.MaxPool2d(kernel_size=3, stride=2)(relu))
        fire3 = self.fire3(fire2)
        fire4 = self.fire4(nn.MaxPool2d(kernel_size=3, stride=2)(fire3))
        fire5 = self.fire5(fire4)
        fire6 = self.fire6(nn.MaxPool2d(kernel_size=3, stride=2)(fire5))
        fire7 = self.fire7(fire6)
        fire8 = self.fire8(fire7)
        fire9 = self.fire9(fire8)
        conv10 = self.conv10(fire9)
        #     downsampling      #
        input_pooling = nn.MaxPool2d(kernel_size=16, stride=8)(input)
        conv1_pooling = nn.MaxPool2d(kernel_size=4, stride=4)(self.conv1(input))
        fire3_pooling = nn.MaxPool2d(kernel_size=2, stride=2)(fire3)
        fire5_pooling = nn.Identity()(fire5)
        conv10_deconv = self.conv10_deconv(conv10)

        #     concatenate & flatten     #
        input_pooling_flatten = torch.flatten(input_pooling)    # torch.Size([2187])
        conv1_pooling_flatten = torch.flatten(conv1_pooling)    # torch.Size([69984])
        fire3_pooling_flatten = torch.flatten(fire3_pooling)    # torch.Size([93312])
        fire5_pooling_flatten = torch.flatten(fire5_pooling)    # torch.Size([186624])
        conv10_deconv_flatten = torch.flatten(conv10_deconv)    # torch.Size([373248])

        fina_x = np.concatenate((input_pooling_flatten.detach().numpy(), conv1_pooling_flatten.detach().numpy(),
                                 fire3_pooling_flatten.detach().numpy(), fire5_pooling_flatten.detach().numpy(),
                                 conv10_deconv_flatten.detach().numpy()))  # print(fina_x.shape)  # (725355,)

        x = torch.tensor(fina_x)

        fc1 = self._fc1(x)
        fc2 = self._fc2(x)
        fc3 = self._fc3(x)
        return fc1, fc2, fc3

    # *****************************************************
    # self.features = nn.Sequential(
    #     self.conv(3, 96, 7, 2),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
    #     Fire(96, 16, 64, 64),
    #     Fire(128, 16, 64, 64),
    #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
    #     Fire(128, 32, 128, 128),
    #     Fire(256, 32, 128, 128),
    #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
    #     Fire(256, 48, 192, 192),
    #     Fire(384, 48, 192, 192),
    #     Fire(384, 64, 256, 256),
    #     Fire(512, 64, 256, 256),
    #     self.conv(512, 512, 3, 2)
    # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = self.extract_features(x)
        # S_x1 = self.Softmax(x1)
        # S_x2 = self.Softmax(x2)
        # S_x3 = self.Softmax(x3)
        # return S_x1, S_x2, S_x3
        return x1, x2, x3


# def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
#     model = SqueezeNet(**kwargs)
#     if pretrained:
#         arch = "squeezenet" + version
#         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def squeezenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
#     r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
#     <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
#     SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
#     than SqueezeNet 1.0, without sacrificing accuracy.
#     The required minimum input size of the model is 17x17.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _squeezenet("1_1", pretrained, progress, **kwargs)


if __name__ == '__main__':
    # 加载模型
    model = SqueezeNet()

    # 获取图片
    image = Image.open(r"D:\TEST_IMAGE\2.jpg")
    image = image.resize((224, 224))
    toTensor = transforms.ToTensor()  # 实例化一个toTensor
    image_tensor = toTensor(image)
    image_tensor = image_tensor.reshape(1, 3, 224, 224)
    output1, output2, output3 = model(image_tensor)
    # # print("model:", model)

    # 获取图片
    # img = cv2.imread(r"D:\TEST_IMAGE\2.jpg")
    # img= cv2.resize(img, (224, 224))
    # toTensor = transforms.ToTensor()
    # img = toTensor(img)
    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img)
    # output1, output2, output3 = model(img)
    # print("output1:", output1)
    # print("output2:", output2)
    # print("output3:", output3)

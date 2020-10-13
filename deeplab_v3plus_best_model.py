import os
import zipfile
import glob
import os
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score
import math
from PIL import Image
import torch
import cv2
import json
from torch import nn
from torch.utils import data
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from collections import namedtuple

annZipFile = "cityscapes/gtFine_trainvaltest.zip"
unZipDir = "cityscapes/gtFine_trainvaltest"
if not os.path.exists(unZipDir):
  print("Unzipping " + annZipFile)
  with zipfile.ZipFile(annZipFile, "r") as zip_ref:
    zip_ref.extractall(unZipDir)
  print("... done unzipping")

annZipFile = "cityscapes/leftImg8bit_trainvaltest.zip"
unZipDir = "cityscapes/leftImg8bit"
if not os.path.exists(unZipDir):
  print("Unzipping " + annZipFile)
  with zipfile.ZipFile(annZipFile, "r") as zip_ref:
    zip_ref.extractall(unZipDir)
  print("... done unzipping")


  def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """

    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


class Cityscapes(data.Dataset):
  """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

  **Parameters:**
      - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
      - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
      - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
      - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
      - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
  """

  # Based on https://github.com/mcordts/cityscapesScripts
  CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                   'has_instances', 'ignore_in_eval', 'color'])
  classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
  ]

  train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
  train_id_to_color.append([0, 0, 0])
  train_id_to_color = np.array(train_id_to_color)
  id_to_train_id = np.array([c.train_id for c in classes])

  # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
  #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
  # train_id_to_color = np.array(train_id_to_color)
  # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

  def __init__(self, root='cityscapes', split='train', mode='fine', target_type='semantic', transform=None):
    self.root = os.path.expanduser(root)
    self.mode = 'gtFine_trainvaltest'
    self.camera = 'camera_trainvaltest'
    self.target_type = target_type
    self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
    self.targets_dir = os.path.join(self.root, self.mode, split)
    self.param_dir = os.path.join(self.root, self.camera, split)

    self.transform = transform

    self.split = split
    self.images = []
    self.targets = []
    self.param = []

    if split not in ['train', 'test', 'val']:
      raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                       ' or split="val"')

    if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
      raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                         ' specified "split" and "mode" are inside the "root" directory')

    for city in os.listdir(self.images_dir):
      img_dir = os.path.join(self.images_dir, city)
      target_dir = os.path.join(self.targets_dir, city)
      param_dir = os.path.join(self.param_dir, city)
      for file_name in os.listdir(img_dir):
        self.images.append(os.path.join(img_dir, file_name))
        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                     self._get_target_suffix(self.mode.split('_')[0], self.target_type))
        param_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'camera.json')

        self.targets.append(os.path.join(target_dir, target_name))

        self.param.append(os.path.join(param_dir, param_name))

  @classmethod
  def encode_target(cls, target):
    return cls.id_to_train_id[np.array(target)]

  @classmethod
  def decode_target(cls, target):
    target[target == 255] = 19
    # target = target.astype('uint8') + 1
    return cls.train_id_to_color[target]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
        than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
    """
    image = Image.open(self.images[index]).convert('RGB')
    target = Image.open(self.targets[index])
    param = self._load_json(self.param[index])
    if self.transform:
      image, target = self.transform(image, target)
    target = self.encode_target(target)
    return image, target, param

  def __len__(self):
    return len(self.images)

  def _load_json(self, path):
    with open(path, 'r') as file:
      data = json.load(file)
    return data

  def _get_target_suffix(self, mode, target_type):
    if target_type == 'instance':
      return '{}_instanceIds.png'.format(mode)
    elif target_type == 'semantic':
      return '{}_labelIds.png'.format(mode)
    elif target_type == 'color':
      return '{}_color.png'.format(mode)
    elif target_type == 'polygon':
      return '{}_polygons.json'.format(mode)
    elif target_type == 'depth':
      return '{}_disparity.png'.format(mode)


annZipFile = "cityscapes/camera_trainvaltest.zip"
unZipDir = "cityscapes/camera_trainvaltest"
if not os.path.exists(unZipDir):
  print("Unzipping " + annZipFile)
  with zipfile.ZipFile(annZipFile, "r") as zip_ref:
    zip_ref.extractall(unZipDir)
  print("... done unzipping")

##########
# TODO: design your own network here. The expectation is to write from scratch. But it's okay to get some inspiration
# from conference paper. The bottom line is that you will not just copy code from other repo
##########
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.utils import load_state_dict_from_url
from collections import OrderedDict

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
  'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

TOTAL_CLASSES = 1


class Flatten(nn.Module):
  def forward(self, x):
    x = x.view(x.size()[0], -1)
    return x


class Vgg16(nn.Module):
  def __init__(self, pretrained=True):
    super(Vgg16, self).__init__()
    self.net = models.vgg16(pretrained).features.eval()

  def forward(self, x):
    out = []
    for i in range(len(self.net)):
      x = self.net[i](x)
      if i in [3, 8, 15, 22, 29]:
        out.append(x)
    return out


class ResNet18(nn.Module):
  def __init__(self, pretrained=True):
    super(ResNet18, self).__init__()
    self.net = models.resnet18(pretrained)

  def forward(self, x):
    out = {}
    for name, module in self.net._modules.items():
      if name == "fc":
        x = x.view(x.size(0), -1)
      x = module(x)
      out[name] = x
    return out


def _make_divisible(v, divisor, min_value=None):
  """
  This function is taken from the original tf repo.
  It ensures that all layers have a channel number that is divisible by 8
  It can be seen here:
  https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
  :param v:
  :param divisor:
  :param min_value:
  :return:
  """
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


class _SimpleSegmentationModel(nn.Module):
  def __init__(self, backbone, classifier):
    super(_SimpleSegmentationModel, self).__init__()
    self.backbone = backbone
    self.classifier = classifier

  def forward(self, x):
    input_shape = x.shape[-2:]
    features = self.backbone(x)
    x = self.classifier(features)
    x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
    return x


class IntermediateLayerGetter(nn.ModuleDict):
  """
  Module wrapper that returns intermediate layers from a model
  It has a strong assumption that the modules have been registered
  into the model in the same order as they are used.
  This means that one should **not** reuse the same nn.Module
  twice in the forward if you want this to work.
  Additionally, it is only able to query submodules that are directly
  assigned to the model. So if `model` is passed, `model.feature1` can
  be returned, but not `model.feature1.layer2`.
  Arguments:
      model (nn.Module): model on which we will extract the features
      return_layers (Dict[name, new_name]): a dict containing the names
          of the modules for which the activations will be returned as
          the key of the dict, and the value of the dict is the name
          of the returned activation (which the user can specify).
  Examples::
      >>> m = torchvision.models.resnet18(pretrained=True)
      >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
      >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
      >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
      >>> out = new_m(torch.rand(1, 3, 224, 224))
      >>> print([(k, v.shape) for k, v in out.items()])
      >>>     [('feat1', torch.Size([1, 64, 56, 56])),
      >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
  """

  def __init__(self, model, return_layers):
    if not set(return_layers).issubset([name for name, _ in model.named_children()]):
      raise ValueError("return_layers are not present in model")

    orig_return_layers = return_layers
    return_layers = {k: v for k, v in return_layers.items()}
    layers = OrderedDict()
    for name, module in model.named_children():
      layers[name] = module
      if name in return_layers:
        del return_layers[name]
      if not return_layers:
        break

    super(IntermediateLayerGetter, self).__init__(layers)
    self.return_layers = orig_return_layers

  def forward(self, x):
    out = OrderedDict()
    for name, module in self.named_children():
      x = module(x)
      if name in self.return_layers:
        out_name = self.return_layers[name]
        print(out_name)
        out[out_name] = x
    return out


class ConvBNReLU(nn.Sequential):
  def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
    # padding = (kernel_size - 1) // 2
    super(ConvBNReLU, self).__init__(
      nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False),
      nn.BatchNorm2d(out_planes),
      nn.ReLU6(inplace=True)
    )


def fixed_padding(kernel_size, dilation):
  kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  return (pad_beg, pad_end, pad_beg, pad_end)


class InvertedResidual(nn.Module):
  def __init__(self, inp, oup, stride, dilation, expand_ratio):
    super(InvertedResidual, self).__init__()
    self.stride = stride
    assert stride in [1, 2]

    hidden_dim = int(round(inp * expand_ratio))
    self.use_res_connect = self.stride == 1 and inp == oup

    layers = []
    if expand_ratio != 1:
      # pw
      layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

    layers.extend([
      # dw
      ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
      # pw-linear
      nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
      nn.BatchNorm2d(oup),
    ])
    self.conv = nn.Sequential(*layers)

    self.input_padding = fixed_padding(3, dilation)

  def forward(self, x):
    x_pad = F.pad(x, self.input_padding)
    if self.use_res_connect:
      return x + self.conv(x_pad)
    else:
      return self.conv(x_pad)


class MobileNetV2(nn.Module):
  def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None,
               round_nearest=8):
    """
    MobileNet V2 main class
    Args:
        num_classes (int): Number of classes
        width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
        inverted_residual_setting: Network structure
        round_nearest (int): Round the number of channels in each layer to be a multiple of this number
        Set to 1 to turn off rounding
    """
    super(MobileNetV2, self).__init__()
    block = InvertedResidual
    input_channel = 32
    last_channel = 1280
    self.output_stride = output_stride
    current_stride = 1
    if inverted_residual_setting is None:
      inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
      ]

    # only check the first element, assuming user knows t,c,n,s are required
    if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
      raise ValueError("inverted_residual_setting should be non-empty "
                       "or a 4-element list, got {}".format(inverted_residual_setting))

    # building first layer
    input_channel = _make_divisible(input_channel * width_mult, round_nearest)
    self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
    features = [ConvBNReLU(3, input_channel, stride=2)]
    current_stride *= 2
    dilation = 1
    previous_dilation = 1

    # building inverted residual blocks
    for t, c, n, s in inverted_residual_setting:
      output_channel = _make_divisible(c * width_mult, round_nearest)
      previous_dilation = dilation
      if current_stride == output_stride:
        stride = 1
        dilation *= s
      else:
        stride = s
        current_stride *= s
      output_channel = int(c * width_mult)

      for i in range(n):
        if i == 0:
          features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
        else:
          features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
        input_channel = output_channel
    # building last several layers
    features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
    # make it nn.Sequential
    self.features = nn.Sequential(*features)

    # building classifier
    self.classifier = nn.Sequential(
      nn.Dropout(0.2),
      nn.Linear(self.last_channel, num_classes),
    )

    # weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)

  def forward(self, x):
    x = self.features(x)
    x = x.mean([2, 3])
    x = self.classifier(x)
    return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
  """
  Constructs a MobileNetV2 architecture from
  `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
      progress (bool): If True, displays a progress bar of the download to stderr
  """
  model = MobileNetV2(**kwargs)
  if pretrained:
    state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
    model.load_state_dict(state_dict)
  return model


class AtrousSeparableConvolution(nn.Module):
  """ Atrous Separable Convolution
  """

  def __init__(self, in_channels, out_channels, kernel_size,
               stride=1, padding=0, dilation=1, bias=True):
    super(AtrousSeparableConvolution, self).__init__()
    self.body = nn.Sequential(
      # Separable Conv
      nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                bias=bias, groups=in_channels),
      # PointWise Conv
      nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
    )

    self._init_weight()

  def forward(self, x):
    return self.body(x)

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
  def __init__(self, in_channels, out_channels, dilation):
    modules = [
      nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    ]
    super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
  def __init__(self, in_channels, out_channels):
    super(ASPPPooling, self).__init__(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(in_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True))

  def forward(self, x):
    size = x.shape[-2:]
    x = super(ASPPPooling, self).forward(x)
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
  def __init__(self, in_channels, atrous_rates):
    super(ASPP, self).__init__()
    out_channels = 256
    modules = []
    modules.append(nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)))

    rate1, rate2, rate3 = tuple(atrous_rates)
    modules.append(ASPPConv(in_channels, out_channels, rate1))
    modules.append(ASPPConv(in_channels, out_channels, rate2))
    modules.append(ASPPConv(in_channels, out_channels, rate3))
    modules.append(ASPPPooling(in_channels, out_channels))

    self.convs = nn.ModuleList(modules)

    self.project = nn.Sequential(
      nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Dropout(0.1), )

  def forward(self, x):
    res = []
    for conv in self.convs:
      res.append(conv(x))
    res = torch.cat(res, dim=1)
    return self.project(res)


def convert_to_separable_conv(module):
  new_module = module
  if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
    new_module = AtrousSeparableConvolution(module.in_channels,
                                            module.out_channels,
                                            module.kernel_size,
                                            module.stride,
                                            module.padding,
                                            module.dilation,
                                            module.bias)
  for name, child in module.named_children():
    new_module.add_module(name, convert_to_separable_conv(child))
  return new_module


class DeepLabHeadV3Plus(nn.Module):
  def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
    super(DeepLabHeadV3Plus, self).__init__()
    self.project = nn.Sequential(
      nn.Conv2d(low_level_channels, 48, 1, bias=False),
      nn.BatchNorm2d(48),
      nn.ReLU(inplace=True),
    )

    self.aspp = ASPP(in_channels, aspp_dilate)

    self.classifier = nn.Sequential(
      nn.Conv2d(304, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, num_classes, 1)
    )
    self._init_weight()

  def forward(self, feature):
    low_level_feature = self.project(feature['low_level'])
    output_feature = self.aspp(feature['out'])
    output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                   align_corners=False)
    return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class DeepLabV3(_SimpleSegmentationModel):
  """
  Implements DeepLabV3 model from
  `"Rethinking Atrous Convolution for Semantic Image Segmentation"
  <https://arxiv.org/abs/1706.05587>`_.
  Arguments:
      backbone (nn.Module): the network used to compute the features for the model.
          The backbone should return an OrderedDict[Tensor], with the key being
          "out" for the last feature map used, and "aux" if an auxiliary classifier
          is used.
      classifier (nn.Module): module that takes the "out" element returned from
          the backbone and returns a dense prediction.
      aux_classifier (nn.Module, optional): auxiliary classifier used during training
  """
  pass

  def segmentation_eval(gts, preds, classes, plot_file_name):
    """
    @param    gts               numpy.ndarray   ground truth labels
    @param    preds             numpy.ndarray   predicted labels
    @param    classes           string          class names
    @param    plot_file_name    string          plot file names
    """
    ious, counts = compute_confusion_matrix(gts, preds)
    aps = compute_ap(gts, preds)
    plot_results(counts, ious, aps, classes, plot_file_name)
    for i in range(len(classes)):
      print('{:>20s}: AP: {:0.2f}, IoU: {:0.2f}'.format(classes[i], aps[i], ious[i]))
    print('{:>20s}: AP: {:0.2f}, IoU: {:0.2f}'.format('mean', np.mean(aps), np.mean(ious)))
    return aps, ious


def plot_results(counts, ious, aps, classes, file_name):
  fig, ax = plt.subplots(1, 1)
  conf = counts / np.sum(counts, 1, keepdims=True)
  conf = np.concatenate([conf, np.array(aps).reshape(-1, 1),
                         np.array(ious).reshape(-1, 1)], 1)
  conf = conf * 100.
  sns.heatmap(conf, annot=True, ax=ax, fmt='3.0f')
  arts = []
  # labels, title and ticks
  _ = ax.set_xlabel('Predicted labels')
  arts.append(_)
  _ = ax.set_ylabel('True labels')
  arts.append(_)
  _ = ax.set_title('Confusion Matrix, mAP: {:5.1f}, mIoU: {:5.1f}'.format(
    np.mean(aps) * 100., np.mean(ious) * 100.))
  arts.append(_)
  _ = ax.xaxis.set_ticklabels(classes + ['AP', 'IoU'], rotation=90)
  arts.append(_)
  _ = ax.yaxis.set_ticklabels(classes, rotation=0)
  arts.append(_)
  fig.savefig(file_name, bbox_inches='tight')


def compute_ap(gts, preds):
  aps = []
  for i in range(preds.shape[1]):
    ap, prec, rec = calc_pr(gts == i, preds[:, i:i + 1, :, :])
    aps.append(ap)
  return aps


def calc_pr(gt, out, wt=None):
  gt = gt.astype(np.float64).reshape((-1, 1))
  out = out.astype(np.float64).reshape((-1, 1))

  tog = np.concatenate([gt, out], axis=1) * 1.
  ind = np.argsort(tog[:, 1], axis=0)[::-1]
  tog = tog[ind, :]
  cumsumsortgt = np.cumsum(tog[:, 0])
  cumsumsortwt = np.cumsum(tog[:, 0] - tog[:, 0] + 1)
  prec = cumsumsortgt / cumsumsortwt
  rec = cumsumsortgt / np.sum(tog[:, 0])
  ap = voc_ap(rec, prec)
  return ap, rec, prec


def voc_ap(rec, prec):
  rec = rec.reshape((-1, 1))
  prec = prec.reshape((-1, 1))
  z = np.zeros((1, 1))
  o = np.ones((1, 1))
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))

  mpre = np.maximum.accumulate(mpre[::-1])[::-1]
  I = np.where(mrec[1:] != mrec[0:-1])[0] + 1;
  ap = np.sum((mrec[I] - mrec[I - 1]) * mpre[I])
  return ap


def compute_confusion_matrix(gts, preds):
  preds_cls = np.argmax(preds, 1)
  gts = gts[:, 0, :, :]
  conf = confusion_matrix(gts.ravel(), preds_cls.ravel())
  inter = np.diag(conf)
  union = np.sum(conf, 0) + np.sum(conf, 1) - np.diag(conf)
  union = np.maximum(union, 1)
  return inter / union, conf


def camera_calibration(param, pos):
  extrinsic_param = param['extrinsic']
  intrinsic_param = param['intrinsic']
  pitch = extrinsic_param['pitch'].numpy()[0]
  roll = extrinsic_param['roll'].numpy()[0]
  yaw = extrinsic_param['yaw'].numpy()[0]
  x = extrinsic_param['x'].numpy()[0]
  y = extrinsic_param['y'].numpy()[0]
  z = extrinsic_param['z'].numpy()[0]
  fx = intrinsic_param['fx'].numpy()[0]
  fy = intrinsic_param['fy'].numpy()[0]
  u0 = intrinsic_param['u0'].numpy()[0]
  v0 = intrinsic_param['v0'].numpy()[0]
  cy = np.cos(yaw)
  cr = np.cos(roll)
  cp = np.cos(pitch)
  sy = np.sin(yaw)
  sr = np.sin(roll)
  sp = np.sin(pitch)

  R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr]])
  t = np.array([[x],
                [y],
                [z]])
  K = np.array([[fx, 0, u0],
                [0, fy, v0],
                [0, 0, 1]])
  intrinsic_rot = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
  R = R.T
  t = np.matmul(-1 * R, t)
  Rt = np.concatenate((R, t), axis=1)
  K = np.matmul(K, intrinsic_rot)
  KRt = np.matmul(K, Rt)
  pos = pos - KRt[:, 2].ravel()
  KRt = np.delete(KRt, [2], 1)
  world_pos = np.linalg.solve(KRt, pos)
  world_z = 1 / world_pos[2]
  world_y = world_z * world_pos[1]
  world_x = world_z * world_pos[0]
  return [world_x, world_y, world_z]


# Colab has GPUs, you will have to move tensors and models to GPU.
device = torch.device("cuda:0"),



#############
#TODO: initialize your model
num_classes = 19
output_stride = 8
if output_stride==8:
    replace_stride_with_dilation=[False, True, True]
    aspp_dilate = [12, 24, 36]
else:
    replace_stride_with_dilation=[False, False, True]
    aspp_dilate = [6, 12, 18]

backbone = mobilenet_v2(pretrained=True, output_stride=output_stride)


backbone.low_level_features = backbone.features[0:4]
backbone.high_level_features = backbone.features[4:-1]
backbone.features = None
backbone.classifier = None

inplanes = 320
low_level_planes = 24

return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)


#print(backbone.low_level_features)
#resnet18 = ResNet18(pretrained=True)
save_dir = 'new_models'
parentModelName = "deeplab_mobilenetv2_v3_plus"
parentEpoch = 1269
IS_GPU = False
model = DeepLabV3(backbone, classifier)
checkpoint = torch.load(os.path.join(save_dir, parentModelName+'_epoch-'+str(parentEpoch)+'.pth'), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])
if IS_GPU:
  model.to(device)
print(model)


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
  return base_lr * ((1 - float(iter_) / max_iter) ** power)


########################################################################
# TODO: Implement your training cycles, make sure you evaluate on validation
# dataset and compute evaluation metrics every so often.
# You may also want to save models that perform well.
from collections import OrderedDict
from dataloaders import custom_transforms as tr
from dataloaders import ext_transforms as et
import timeit

testBatch = 4  # Testing Batch
EPOCHS = 800
snapshot = 10
nEpochs = EPOCHS
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

# Tune the learning rate.
# See whether the momentum is useful or not
# Use the following optimizer
p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 4  # Training batch size
testBatch = 4  # Testing batch size
useTest = True  # See evolution of the test set when training
nValInterval = 5  # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-7  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 10  # How many epochs to change learning rate
# learning_rate = 1e-8
#     wd = 0.0002
# Setting of parameters
# Parameters in p are used for the name of the model
optimizer = optim.SGD(model.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
p['optimizer'] = str(optimizer)
"""
optimizer = optim.SGD([
    {'params': [pr[1] for pr in model.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd,
     'initial_lr': lr},
    {'params': [pr[1] for pr in model.stages.named_parameters() if 'bias' in pr[0]], 'lr': 2 * lr, 'initial_lr': 2 * lr},
    {'params': [pr[1] for pr in model.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd,
     'initial_lr': lr},
    {'params': [pr[1] for pr in model.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': 2 * lr,
     'initial_lr': 2 * lr},
    {'params': [pr[1] for pr in model.score_dsn.named_parameters() if 'weight' in pr[0]], 'lr': lr / 10,
     'weight_decay': wd, 'initial_lr': lr / 10},
    {'params': [pr[1] for pr in model.score_dsn.named_parameters() if 'bias' in pr[0]], 'lr': 2 * lr / 10,
     'initial_lr': 2 * lr / 10},
    {'params': [pr[1] for pr in model.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0, 'initial_lr': 0},
    {'params': [pr[1] for pr in model.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0, 'initial_lr': 0},
    {'params': net.fuse.weight, 'lr': lr / 100, 'initial_lr': lr / 100, 'weight_decay': wd},
    {'params': net.fuse.bias, 'lr': 2 * lr / 100, 'initial_lr': 2 * lr / 100},
], lr=lr, momentum=0.9)
"""
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=wd)

composed_transforms_tr = et.ExtCompose([
  # et.ExtResize( 512 ),
  et.ExtRandomCrop(size=(513, 513)),
  et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
  et.ExtRandomHorizontalFlip(),
  et.ExtToTensor(),
  et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

composed_transforms_ts = et.ExtCompose([
  et.ExtResize(512),
  # et.ExtRandomCrop(size=(513, 1024)),
  et.ExtToTensor(),
  et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#cityscapes_train = Cityscapes(split='train', transform=composed_transforms_tr)
cityscapes_val = Cityscapes(split='val', transform=composed_transforms_ts)
# cityscapes_test = CityscapesSegmentation(split='test', transform=composed_transforms_ts)


#train_dataloader = data.DataLoader(cityscapes_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)



from collections import namedtuple
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
trainId2label   = { label.trainId : label for label in reversed(labels) }


import timeit
import random
import matplotlib.pyplot as plt

temp = []
def __onclick__(event, param, image_pixel):
  global ix, iy
  ix, iy = event.xdata, event.ydata
  global coords
  if ix and iy:
    coords = np.array([ix, iy, 0.5])
    row = int(image_pixel.shape[0] - 1 - iy)
    column = int(ix)
    world_coords = camera_calibration(param, coords * 2)
    for p in temp:
      p.remove()
      temp.pop()
    points = plt.plot([ix], [iy], 'go')
    temp.extend(points)
    if (image_pixel[row, column] > (image_pixel.mean())):
      print("x=", world_coords[0], ", y=", world_coords[1], ", z", world_coords[2])
    else:
      print("有障碍物")
    plt.show()

def imshow(img):
  img = img / 2.0 + 0.5  # unnormalize
  npimg = img.numpy()
  fig = plt.figure()
  coords = fig.canvas.callbacks.connect('button_press_event', __onclick__)
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


subset_indices = []
for i in range(0, 1):
  n = random.randint(1, 400)
  subset_indices.append(n)

subset = torch.utils.data.Subset(cityscapes_val, subset_indices)
testloader_1 = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

model.eval()
for i, data in enumerate(testloader_1):

  img, gt_label, param = data
  if IS_GPU:
    img = img.cuda()
  out = model(img)
  if IS_GPU:
    img = img.cpu()
    out = out.detach().cpu()
  img = torchvision.utils.make_grid(img)
  out_img = img.numpy().copy()
  alpha = 0.4
  for id in range(out.shape[1]):
    # out = torch.sum(out,dim=1)
    # print()
    Img = np.ones(out_img.shape, out_img.dtype)
    Img[0, :, :] = trainId2label[id].color[0] * Img[0, :, :]
    Img[1, :, :] = trainId2label[id].color[1] * Img[1, :, :]
    Img[2, :, :] = trainId2label[id].color[2] * Img[2, :, :]
    Img = Img / 255.0
    mask_id = torchvision.utils.make_grid(out[:, id, :, :])
    mask_id = mask_id.detach().numpy()
    # mask_id = (mask_id - mask_id.mean()) / mask_id.std()
    # mask_id[mask_id<(mask_id.mean()-mask_id.std())]=0
    # mask_id = mask_id.clip(0,1)
    mask = Img * mask_id
    out_img = cv2.addWeighted(mask, alpha, out_img, 1.0, 0, out_img)
  plot_img = torch.from_numpy(out_img) / 2.0 + 0.5  # unnormalize
  npimg = plot_img.numpy()
  fig = plt.figure()
  coords = fig.canvas.callbacks.connect('button_press_event', lambda event: __onclick__(event, param, out[0, 6, :, :]))
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

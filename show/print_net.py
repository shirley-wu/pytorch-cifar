# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms

import tkinter.filedialog
import matplotlib.pyplot as plt
from PIL import Image

from load import loadnet

net, _ = loadnet(7)

print(net)
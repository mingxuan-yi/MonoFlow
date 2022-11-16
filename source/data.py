import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch import nn
import torch

def get_dataset(dataroot, name):
    transform = transforms.Compose([transforms.ToTensor()])
    if name == 'mnist':
        dataset = MNIST(dataroot, transform=transform, download=True)
    elif name == 'cifar10':
        dataset = CIFAR10(dataroot, transform=transform, download=True)
    return dataset

class DataTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    # Transform (0, 1) to logit transform 
    def __init__(self, alpha, logit_trans=False):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.logit_trans = logit_trans

    def forward_transform(self, x):
        if not self.logit_trans:
            return 2*x - 1.
        
        x = x / 256. * 255. + torch.rand_like(x) / 256.
        return self.logit_transform(x)

    def reverse(self, y, clip=True):
        if not self.logit_trans:
            return y.clip(-1., 1.)
        
        x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if clip:
            return torch.clamp(x, -1.0, 1.0)
        return x

    def logit_transform(self, image: torch.Tensor):
        
        image = self.alpha + (1 - 2 * self.alpha) * image
        return torch.log(image) - torch.log1p(-image)


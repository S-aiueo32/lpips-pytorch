from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image
from torch.testing import assert_allclose

from lpips_pytorch import LPIPS
from lpips_pytorch import lpips
from PerceptualSimilarity.models import PerceptualLoss

img = Image.open(Path(__file__).parents[1].joinpath('data/lenna.png'))
img_x2 = img.resize((x // 2 for x in img.size)).resize(img.size)

tensor_org = TF.to_tensor(img).unsqueeze(0) * 2 - 1
tensor_x2 = TF.to_tensor(img_x2).unsqueeze(0) * 2 - 1


def test_functional():
    assert lpips(tensor_x2, tensor_org)


def test_functional_on_gpu():
    assert lpips(tensor_x2.to('cuda:0'), tensor_org.to('cuda:0'))


def test_on_gpu():
    org_criterion = PerceptualLoss(net='alex', use_gpu=True)
    my_criterion = LPIPS('alex', version='0.1').to('cuda:0')

    org_loss = org_criterion.forward(
        tensor_x2.to('cuda:0'), tensor_org.to('cuda:0'))
    my_loss = my_criterion(
        tensor_x2.to('cuda:0'), tensor_org.to('cuda:0'))

    assert_allclose(org_loss, my_loss)


def test_alex_v0_1():
    org_criterion = PerceptualLoss(net='alex', use_gpu=False)
    my_criterion = LPIPS('alex', version='0.1')

    org_loss = org_criterion.forward(tensor_x2, tensor_org)
    my_loss = my_criterion(tensor_x2, tensor_org)

    assert_allclose(org_loss, my_loss)


def test_squeeze_v0_1():
    org_criterion = PerceptualLoss(net='squeeze', use_gpu=False)
    my_criterion = LPIPS('squeeze', version='0.1')

    org_loss = org_criterion.forward(tensor_x2, tensor_org)
    my_loss = my_criterion(tensor_x2, tensor_org)

    assert_allclose(org_loss, my_loss)


def test_vgg_v0_1():
    org_criterion = PerceptualLoss(net='vgg', use_gpu=False)
    my_criterion = LPIPS('vgg', version='0.1')

    org_loss = org_criterion.forward(tensor_x2, tensor_org)
    my_loss = my_criterion(tensor_x2, tensor_org)

    assert_allclose(org_loss, my_loss)

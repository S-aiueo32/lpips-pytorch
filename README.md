# lpips-pytorch
[![CircleCI](https://circleci.com/gh/S-aiueo32/lpips-pytorch.svg?style=svg)](https://circleci.com/gh/S-aiueo32/lpips-pytorch)  ![](https://img.shields.io/badge/LPIPS%20ver.-0.1-brightgreen)

## Description
Developing perceptual distance metrics is a major topic in recent image processing problems.
LPIPS[1] is a state-of-the-art perceptual metric based on human similarity judgments.
The official implementation is not only publicly available as a metric, but also enables users to train the new metric by themselves.
In other words, The official implementation has less simplicity by the high-level wrapping for training.
This repository provides an alternative simple and useful implementation of LPIPS.
This output is definitely the same result because the weights are converted from the original one.

## Requirement
- `torch` >= 1.3
- `torchvision` >= 0.4

## Usage
```python
from lpips_pytorch import LPIPS, lpips


# define as a criterion module (recommended)
criterion = LPIPS(
    net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)
loss = criterion(x, y)

# functional call
loss = lpips(x, y, net_type='alex', version='0.1')
```

## Install
1. Clone this repository and move into your project
    ```shell
    ~ $ git clone https://github.com/S-aiueo32/lpips-pytorch.git
    ~ $ mv lpips-pytorch/lpips-pytorch <YOUR_PROJECT>
    ```
2. Install by `pip`
    ```shell
    $ pip install git+https://github.com/S-aiueo32/lpips-pytorch.git
    ```

## License
[BSD 2-Clause "Simplified" License](LICENSE).

## Reference
1. Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

## Acknowledgements
This project directly uses the original weights, many thanks to the authors.

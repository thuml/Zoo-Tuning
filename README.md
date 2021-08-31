# Zoo-Tuning

Code release for [Zoo-Tuning: Adaptive Transfer from A Zoo of Models
](https://arxiv.org/abs/2106.15434) (ICML2021)

## Pretrained Models

| Pretrained Models | Reference |
| ------ | ------ |
| ImageNet Supervised	 | https://pytorch.org/vision/stable/models.html#id10 |
| MoCo | https://github.com/facebookresearch/moco |
| Mask R-CNN | https://pytorch.org/vision/stable/models.html#id41 |
| DeepLabV3 | https://pytorch.org/vision/stable/models.html#deeplabv3 |
| Keypoint R-CNN | https://pytorch.org/vision/stable/models.html#keypoint-r-cnn |

For convenience, we also provide the pretrained models downloaded from these pages. [Download](https://drive.google.com/file/d/1407oe8AooagrHNRZNLrbvFxzM_z2D1Wg/view?usp=sharing)

## Datasets

| Dataset | Download Link |
| ------ | ------ |
| CIFAR-100	 | Downloaded automatically from torchvision. |
| COCO-70	 | https://github.com/thuml/CoTuning |
| FGVC Aircraft | http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |
| Stanford Cars | http://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| MIT Indoors | http://web.mit.edu/torralba/www/indoor.html |

## Requirements

* Python 3.8
* PyTorch 1.8.0
* tqdm
* einops
* requests

## Quick Start

* Download and prepare the pretrained models in `pretrained_models/`.
* Download the DATASET you need or prepare your own DATASET. Then change the dataset paths of `get_data_loader()` in `main.py` to the directory of the DATASET.
* We provide the training script, `train.sh`. Complete the configuration of experiments, then `bash train.sh` for training and testing.

## Citation
If you find this code or our paper useful, please consider citing:<br>

```
@inproceedings{shu2021zoo,
  title={Zoo-Tuning: Adaptive Transfer from a Zoo of Models},
  author={Shu, Yang and Kou, Zhi and Cao, Zhangjie and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Machine Learning},
  pages={9626--9637},
  year={2021},
  organization={PMLR}
}
```

## Contact
If you have any problems about our code, feel free to contact<br>

* shu-y18@mails.tsinghua.edu.cn
* kz19@mails.tsinghua.edu.com

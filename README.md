# FourierSR: A Fourier Token-based Plugin for Efficient Image Super-Resolution

### [Paper (ArXiv)](https://arxiv.org/pdf/2503.10043) | [Paper (IEEE)]() | [Supplementary Material](https://drive.google.com/file/d/1f8HkDjl8Hu0rxLFMLW7TzeqKzdQjhpLr/view?usp=sharing)

---
## Contents

The contents of this repository are as follows:

1. [Dependencies](#Dependencies)
2. [Train](#Train)
3. [Test](#Test)

## Dataset
We used only the first 800 images of <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a> dataset to train our model.

The test set including Set5, Set14, B100, Urban100, Manga109, which can be downloaded from <a href="https://pan.baidu.com/s/1Vb68GWERriLmJRtYfm2uEg">here</a>.[Password:8888]

The code and datasets need satisfy the following structures:
```
├── FourierSR  			# Train / Test Code
├── dataset  					# all datasets for this code
|  └── DIV2K_decoded  		#  train datasets with npy format
|  |  └── DIV2K_train_HR  		
|  |  └── DIV2K_train_LR_bicubic 			
|  └── benchmark  		#  test datasets with png format 
|  |  └── Set5
|  |  └── Set14
|  |  └── B100
|  |  └── Urban100
|  |  └── Manga109
 ─────────────────
```
---

## Results

Our SR Results can be downloaded from <a href="https://drive.google.com/file/d/1aiwz11jVFezUed5q-24VCVfNOy0C9ja4/view?usp=drive_link">here</a>.

Pretrained models can be found in <a href="https://drive.google.com/file/d/1VmsmDckdnWg57ntdx8LtIhGh4TPEf6nG/view?usp=drive_link">here</a>.

---

### Dependencies
> - Python >= 3.7
> - torch >= 1.2
> - einops
> - timm
> - tqdm
> - imageio


### Train

```
###  As EDSR as a sample  ###

# For X2
python3 main.py --model edsr_fre --scale 2 --patch_size 96 --extra_loss --save Offical_EDSRfrex2

# For X3
python3 main.py --model edsr_fre --scale 3 --patch_size 144 --extra_loss --save Offical_EDSRfrex3

# For X4
python3 main.py --model edsr_fre --scale 4 --patch_size 192 --extra_loss --save Offical_EDSRfrex4
```

### Test

```
###  As EDSR-FourierSR as a sample  ###

# For X2
python3 main.py --model edsr_fre --scale 2 --data_test Set5+Set14+B100+Urban100+Manga109 --save_results --save test_results/EDSRfrex2_results --pre_train ../experiment/Offical_EDSRfrex2/edsr_fre_x2.pt

# For X3
python3 main.py --model edsr_fre --scale 3 --data_test Set5+Set14+B100+Urban100+Manga109 --save_results --save test_results/EDSRfrex2_results --pre_train ../experiment/Offical_EDSRfrex3/edsr_fre_x3.pt

# For X4
python3 main.py --model edsr_fre --scale 4 --data_test Set5+Set14+B100+Urban100+Manga109 --save_results --save test_results/EDSRfrex2_results --pre_train ../experiment/Offical_EDSRfrex4/edsr_fre_x4.pt
```

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{li2026fouriersr,
  title={FourierSR: A Fourier Token-based Plugin for Efficient Image Super-Resolution},
  author={Li, Wenjie and Guo, Heng and Hou, Yuefeng and Ma, Zhanyu},
  journal={IEEE Transactions on Image Processing},
  year={2026}
}
```

## Acknowledgement

The foundation for the training process is profited from the outstanding contribution of [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch).

## Contact

This repo is currently maintained by lewj2408@gmail.com and is for academic research use only. 

# AI Image Signal Processing

[Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)

Computer Vision Lab, CAIDAS, University of Würzburg

#### Contents

1. [AIM 2022 Reversed ISP Challenge](#aim-2022-reversed-isp-challenge) Official repo!
1. [Model-Based Image Signal Processors via Learnable Dictionaries](https://ojs.aaai.org/index.php/AAAI/article/view/19926) AAAI 2022 Oral
1. [MAI 2022 Learned ISP Challenge](#mai-2022-learned-isp-challenge) Complete Baseline solution
1. [Citation and Acknowledgement](#citation-and-acknowledgement)
1. [Contact](#contact)

---------------------------------------------------


## [AIM 2022 Reversed ISP Challenge](https://data.vision.ee.ethz.ch/cvl/aim22/) 

### [Track 1 - S7](https://codalab.lisn.upsaclay.fr/competitions/5079)
### [Track 2 - P20](https://codalab.lisn.upsaclay.fr/competitions/5080)

<a href="https://data.vision.ee.ethz.ch/cvl/aim22/"><img src="https://i.ibb.co/VJ7SSQj/aim-challenge-teaser.png" alt="aim-challenge-teaser" width="400" border="0"></a>

Digital cameras transform sensor RAW readings into RGB images by means of their Image Signal Processor (ISP). Computational photography tasks such as image denoising and colour constancy are commonly performed in the RAW domain, in part due to the inherent hardware design, but also due to the appealing simplicity of noise statistics that result from the direct sensor readings. Despite this, the availability of RAW images is limited in comparison with the abundance and diversity of available RGB data. Recent approaches have attempted to bridge this gap by estimating the RGB to RAW mapping [1-5].

In this challenge, we look for solutions to recover RAW readings from the camera using only the corresponding RGB images processed by the in-camera ISP. Successful solutions should generate plausible RAW images, and by doing this, other downstream tasks like Denoising, Super-resolution or Colour Constancy can benefit from such synthetic data generation.

### Starter guide and code 🔥

- **[aim-starter-code.ipynb](aim-starter-code.ipynb)** - Simple dataloading and visualization of RGB-RAW pairs + other utils.
- **[aim-baseline.ipynb](aim-baseline.ipynb)** - End-to-end guide to load the data, train a simple UNet model and make your first submission!

We also provide the baseline pretrained [starter-models](starter-models) and other utils like the scoring scripts.
You can download here a [track 1](https://drive.google.com/file/d/1RmIqB48jeS6HZuf2x2_DUqI9IssOEWNI/view?usp=sharing) and [track 2](https://drive.google.com/file/d/1AlG3WbZtCztpgMkw5S3cAwZYOyU5Wpyq/view?usp=sharing) dummy submissions.
If you want to double-check your submission please check [scoring/evaluation.py](scoring/evaluation.py), it is important to save properly the RAW images as indicated. Read more details below.

<a href="[https://ibb.co/ckVNydX](https://codalab.lisn.upsaclay.fr/competitions/5079)"><img src="https://i.ibb.co/hLwKgv1/Screenshot-from-2022-06-11-22-33-14.png" alt="Screenshot-from-2022-06-11-22-33-14" width="450" border="0"></a>


### Datasets

**Samsung S7 DeepISP Dataset**

We use a custom version of the dataset collected by Schwartz et al. in their work *[DeepISP: Learning End-to-End Image Processing Pipeline](https://arxiv.org/abs/1801.06724)*. We process the original RAW images (GRBG pattern) and extract "aligned" RGB-RAW pairs (patches).

<img src="media/sample2.jpg" width="500" alt="" border="0">


**ETH Huawei P20 Dataset**

We use a custom version of the dataset from [ETH PyNET by Ignatov et al.](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset), a large-scale dataset consisting of RAW-RGB image pairs captured in the wild with the Huawei P20 camera (12.3 MP Sony Exmor IMX380 sensor).
More information in their paper *[Replacing Mobile Camera ISP with a Single Deep Learning Model](https://arxiv.org/abs/2002.05509)*

<img src="media/sample1.jpg" width="500" alt="" border="0">

**Data Structure**

In both tracks, we provide RAW and RGB images extracted from the camera ISP. 
- RAW images are provided in `.npy` format, as 4-channel images following the RGGB pattern
- RGB images are provided in `.jpg` format

You can download the competition data after registering at the challenge [here](https://codalab.lisn.upsaclay.fr/competitions/5079).

The data sctructure should be as follows:

```
├── data-p20
│   ├── train
│   └── val_rgb
└── data-s7
    ├── train
    └── val_rgb
```

`data-p20` should be around 3 Gb and `data-s7` should be around 2.5 Gb

- `train/` contains RGBs in `.jpg` format and RAWs in `.npy` format
- `val_rgb/` contains RGBs in `.jpg`

From the corresponding RGBs, you need to reverse the ISP operations an provide the corresponding RAW image.

For each track, the training / validation split is as follows:
- track 1:  Training samples: 4320 	 Validation samples: 480
- track 2:  Training samples: 5760 	 Validation samples: 720

### Hints and Tips

- RAW images are provided as `np.uint16` with max value `2**10 = 1024`. 
- The RAW images are packed as (h,w,4) , you can unpack it and obtain a (h*2, w*2,1) RAW using the corresponding utils. We recommend to use the 4-channel RAW image.
- RAW images are already converted to standard RGGB pattern.
- Mosaic and Demosaic functions are provided for visualization purposes only.
- For the S7 dataset, most of the images are well-aligned, SSIM and PSNR should work as perceptual metrics.
- For the HP20 dataset, most of the images are **not** aligned. The RGB from the ISP is the process of many transformations including cropping and zooming. Therefore, in this track we recommend perceptual losses as SSIM, MSSSIM and KL-Divergence. In this track, we focus on SSIM as standard metric, but we will consider internally the other mentioned metrics. 
- The ISP corrects many artifacts such as noise and blur. The original RAW images threfore might contain such artifacts.

------

## [MAI 2022 Learned ISP Challenge](https://codalab.lisn.upsaclay.fr/competitions/1759)

You can find at [mai22-learnedisp](mai22-learnedisp/) and end-to-end baseline: dataloading, training top solution, model conversion to tflite.
The model achieved 23.46dB PSNR after training for a few hours. Here you can see a sample RAW input and the resultant RGB.

<img src="mai22-learnedisp/result-isp3.png" width="400" border="0">

We test the model on AI Benchmark. The model average latency is 60ms using a input RAW image `544,960,4` and generating a RGB `1088,1920,3`, in a mid-level smartphone (45.4 AI-score) using Delegate GPU and FP16.


------

## Related Work

[1] [Model-Based Image Signal Processors via Learnable Dictionaries](https://ojs.aaai.org/index.php/AAAI/article/view/19926) by Conde et al, AAAI 2022.

[2] [Learned Smartphone ISP on Mobile NPUs with Deep Learning, Mobile AI 2021 Challenge: Report](https://arxiv.org/abs/2105.07809) by Ignatov et al, CVPRW 2021.

[3] [Learning Raw Image Reconstruction-Aware Deep Image Compressors](https://abhijithpunnappurath.github.io/pami_raw.pdf) by Abhijith Punnappurath and Michael S. Brown, TPAMI 2019.

[4] [Unprocessing Images for Learned Raw Denoising](https://arxiv.org/abs/1811.11127) by Brooks et al. , CVPR 2019

[5] [CycleISP: Real Image Restoration via Improved Data Synthesis](https://arxiv.org/abs/2003.07761) by Zamir et al. , CVPR 2020

**Related Challenge**

[Mobile AI & AIM 2022 Learned Smartphone ISP Challenge](https://codalab.lisn.upsaclay.fr/competitions/1759) organized by Andrey Ignatov.


-----------------

## Citation and Acknowledgement

```
@article{conde2022modelbased, 
	title={Model-Based Image Signal Processors via Learnable Dictionaries},
	author={Conde, Marcos V. and McDonagh, Steven and Maggioni, Matteo and Leonardis, Ales and Pérez-Pellitero, Eduardo},
	journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
	url={https://ojs.aaai.org/index.php/AAAI/article/view/19926}, 
	DOI={10.1609/aaai.v36i1.19926},  
	year={2022}, 
	month={Jun.},
	volume={36},
	number={1},  
	pages={481-489} 
}
```


## Contact

Marcos Conde (marcos.conde-osorio@uni-wuerzburg.de) and Radu Timofte (radu.timofte@uni-wuerzburg.de) are the contact persons and direct managers of the AIM challenge. Please add in the email subject "AIM22 Reverse ISP Challenge" or "AISP"

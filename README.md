# USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection


Source code and dataset for our paper “USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection （IEEE T-IP 2023）” by Lin Hong,  Xin Wang, Gan Zhang and Ming Zhao.

Created by Lin Hong, email: 20B953023@stu.hit.edu.cn

## USOD10K dataset
Baidu Netdisk: [USOD10K](https://pan.baidu.com/s/15sXImJJooDfPF-0cTA6kIg) fetch code: [ceic]  &&&  Google drive:[USOD10K](https://drive.google.com/drive/folders/1ckfPAiMHg0cs9ShV835JV_93jlEnTrKG?usp=share_link) is the first large-scale dataset for Underwater Salient Object Detection (USOD). It is free for academic research, not for any commercial purposes.

## I will try my best to upload all the code and predicted saliency maps this month (July), please give me some patience. =:)

For practical training and reliable test results of deep methods on the USOD10K dataset, there should be enough samples of each category on the training set, validation set, and test set. Hence we follow the USOD10K split of roughly 7:2:1. Its folder looks like this:

````
   Data
   |-- USOD10K
   |   |-- USOD10K-TR
   |   |-- |-- USOD10K-TR-RGB
   |   |-- |-- USOD10K-TR-GT
   |   |-- |-- USOD10K-TR-depth
   |   |-- |-- USOD10K-TR-Boundary
   |   |-- USOD10K-Val
   |   |-- |-- USOD10K-Val-RGB
   |   |-- |-- USOD10K-Val-GT
   |   |-- |-- USOD10K-Val-depth
   |   |-- |-- USOD10K-Val-Boundary
   |   |-- USOD10K-TE
   |   |-- |-- USOD10K-TE-RGB
   |   |-- |-- USOD10K-TE-GT
   |   |-- |-- USOD10K-TE-depth
````
## TC-USOD baseline
![](TC-USOD.png)
### Requirement
1. Python 3.8
1. Pytorch 1.6.0
2. Torchvison 0.7.0

### Our source code will released soon. !!!!


## Benchmark
We retrained 35 SOTA methods, most of the deep methods are proposed in the year of 2020 and 2021. It takes us about 1750 hours to retrain these methods.

Retrained model are available [benchmark_pth](https://pan.baidu.com/s/1C20F8q_LRGTRLQPwcX9Gsw?pwd=qex5) fetch code: [qex5]

Predicted saliency maps [USOD10K_predictions](https://pan.baidu.com/s/16tirqN1X5xXODzIZbSzCRA?pwd=q13c) fetch code: [q13c]

Predicted saliency maps [USOD_predictions](https://pan.baidu.com/s/1Mq8ib52um1ZN7fFl1nWFLg?pwd=ygbg) fetch code: [ygbg]  

[Evaluation results](https://pan.baidu.com/s/14ui8-FLBi0WM58VhEwdzsw?pwd=cqgh) fetch code: [cqgh]  

## Acknowledgement
We thank the authors of [VST](https://github.com/yitu-opensource/T2T-ViT) for providing T2T-ViT backbone, the authors of [DPT](https://github.com/isl-org/DPT) for providing us the method to get estimated depth maps of single underwater images in USOD10K, the authors of [SVAM-Net](http://www.roboticsproceedings.org/rss18/p048.pdf) for providing the [USOD](https://irvlab.cs.umn.edu/resources/usod-dataset) dataset, and [Zhao Zhang](https://github.com/zzhanghub/eval-co-sod) for providing the efficient evaluation tool.




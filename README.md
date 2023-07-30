# USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection

Source code and dataset for our paper “USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection paper” by Lin Hong,  Xin Wang, Gan Zhang, and Ming Zhao. [IEEE TIP 2023](https://ieeexplore.ieee.org/document/10102831)

Created by Lin Hong, email: 20B953023@stu.hit.edu.cn

## USOD10K dataset
Baidu Netdisk: [USOD10K](https://pan.baidu.com/s/11icEoJRqKmStkUkEtRgS4w) fetch code: [yom5]  &&&  Google drive: [USOD10K](https://drive.google.com/file/d/1PH0PwKchXnkWwtAwbhNSW4utMCp5zer8/view?usp=sharing) is the first large-scale dataset for Underwater Salient Object Detection (USOD). It is free for academic research, not for any commercial purposes.

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

## Benchmark
We retrained 35 SOTA methods, most of the deep methods are proposed in the year of 2020 and 2021. It takes us about 1750 hours to retrain these methods.

Retrained model are available [benchmark_pth](https://pan.baidu.com/s/1N4bMHCsvLrHckgv4RFODyQ) fetch code: [sdae]

Predicted saliency maps [USOD10K_predictions](https://pan.baidu.com/s/16tirqN1X5xXODzIZbSzCRA?pwd=q13c) fetch code: [q13c]

Predicted saliency maps [USOD_predictions](https://pan.baidu.com/s/1pdFUEbKMllWFXxBbJ0JayA) fetch code: [niwo]  

[Evaluation results](https://pan.baidu.com/s/1Z4XdgNpcwWi7UKyDnoZqdQ) fetch code: [kivt]  

## Acknowledgement
We thank the authors of [VST](https://github.com/yitu-opensource/T2T-ViT) for providing T2T-ViT backbone, the authors of [DPT](https://github.com/isl-org/DPT) for providing us the method to get estimated depth maps of single underwater images in USOD10K, the authors of [SVAM-Net](http://www.roboticsproceedings.org/rss18/p048.pdf) for providing the [USOD](https://irvlab.cs.umn.edu/resources/usod-dataset) dataset, and [Zhao Zhang](https://github.com/zzhanghub/eval-co-sod) for providing the efficient evaluation tool.




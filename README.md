# USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection

Source code and dataset for our paper “**USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection paper**” by Lin Hong,  Xin Wang, Gan Zhang, and Ming Zhao. [IEEE TIP 2023](https://ieeexplore.ieee.org/document/10102831)

Created by **Lin Hong**, email: 20B953023@stu.hit.edu.cn

## USOD10K dataset
Baidu Netdisk: [USOD10K](https://pan.baidu.com/s/1edg2B9HjnHdEpmwnUOT0-w) **fetch code**: [good]  &&&  Google drive: [USOD10K](https://drive.google.com/file/d/1PH0PwKchXnkWwtAwbhNSW4utMCp5zer8/view?usp=sharing) is the first large-scale dataset for Underwater Salient Object Detection (USOD). It is free for academic research, not for any commercial purposes.

![avatar](https://github.com/LinHong-HIT/USOD10K/blob/ef4fc30f7957f3255e375b608191175454cf4658/dataset_img.png)
Note: for practical training and reliable test results of deep methods on the USOD10K dataset, there should be enough samples of each category on the training set, validation set (**training set and validation set are merged in TC-USOD baseline**), and test set. Hence we follow the USOD10K split of roughly 7:2:1. Its folder looks like this:

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
The TC-USOD baseline is simple but strong, it adopts a hybrid architecture based on an encoder-decoder design that leverages transformer and convolution as the basic computational building block of the encoder and decoder, respectively. 

**How to generate predicted saliency maps by yourself or retrain this model:**
You create a folder named checkpoint under the TU_USOD folder (cd TC_USOD->mkdir checkpoint) and put the [TC-USOD baseline](https://pan.baidu.com/s/1TwwaTcdmTiU2FHOC5xC3Vw) **fetch code**: [ie0k] in it to generate the predicted saliency maps (**you can also find them in the TC_USOD/preds/USOD10K in this project**). Of course, you can retrain this method with the available USOD10K dataset to get your own model. 

![](TC-USOD.png)
### Requirement
1. Python 3.8
2. Pytorch 1.6.0
3. Torchvison 0.7.0

## Benchmark
We retrained 35 SOTA methods in the fields of SOD and USOD, most of the deep methods are proposed in the years 2020, 2021, and 2022. It takes us about 1750 hours to retrain these methods. Here is the qualitative evaluation of the 35 SOTA methods and the TC-USOD baseline.
![avatar](https://github.com/LinHong-HIT/USOD10K/blob/4ac87c771709fc62ba0bce219cdaec2bee176c0d/qualitative_eva.png)

(1) Retrained models are available [benchmark_pth](https://pan.baidu.com/s/1VXyNHxy5Iy5GYYBCh_2thg) **fetch code**: [usod]

(2) Predicted saliency maps [USOD10K_predictions](https://pan.baidu.com/s/1EpnE07lgamyaUIUZWdccqA) **fetch code**: [usod]

(3) Predicted saliency maps [USOD_predictions](https://pan.baidu.com/s/1cnmMZ0JSshssm2jc9p2BdA ) **fetch code**: [usod]  

(4) [Evaluation results](https://pan.baidu.com/s/1AL4WQeFh1KrD0jj9JW182g) **fetch code**: [usod]  

## Bibliography entry
If you think our work is helpful, please cite
```
@ARTICLE{10102831,
  author={Hong, Lin and Wang, Xin and Zhang, Gan and Zhao, Ming},
  journal={IEEE Transactions on Image Processing}, 
  title={USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2023.3266163}}
```

## SOD dataset

(1) NJUD [[baidu pan](https://pan.baidu.com/s/1ywIJV_C0lG1KZNFow87bQQ) fetch code: 7mrn | [Google drive](https://drive.google.com/file/d/19rdcNsuDE6bRD58bruqCXPDhoopTMME4/view?usp=sharing)]  
(2) NLPR [[baidu pan](https://pan.baidu.com/s/1G3ec34XV7oQboY8R9FPVDw) fetch code: tqqm | [Google drive](https://drive.google.com/file/d/1NlJqeauFt6NlzNSHL9iQofzm8XWLmeg9/view?usp=sharing)]  
(3) DUTLF-Depth [[baidu pan](https://pan.baidu.com/s/1BZepaCfo2BsuvBczJKhN4Q) fetch code: 9jac | [Google drive](https://drive.google.com/file/d/1FcS2cBrIj-tBmEgqQzqp-arKIA6UjsLd/view?usp=sharing)]  
(4) STERE [[baidu pan](https://pan.baidu.com/s/16ros8tHMxy9YwfqBZJf1zQ) fetch code: 93hl | [Google drive](https://drive.google.com/file/d/1cVw3tM3xRBxrvO3TZ-oX5tmnPPMIrNbJ/view?usp=sharing)]  
(5) LFSD [[baidu pan](https://pan.baidu.com/s/1sSjFX45DIcNyExsA_lpybQ) fetch code: l2g4 | [Google drive](https://drive.google.com/file/d/1KFZ53EiIuCxMaf6nlFwhfOeBqOJ7BldF/view?usp=sharing)]  
(6) RGBD135 [[baidu pan](https://pan.baidu.com/s/1NQiTSYIs23Cl4TCf7Edp0A) fetch code: apzb | [Google drive](https://drive.google.com/file/d/1kYClZ_17EdFviJ6SiW0_ghqudUCr4r2F/view?usp=sharing)]  
(7) SSD [[baidu pan](https://pan.baidu.com/s/1Ihx001o1MUYaUtbBQH4TnQ) fetch code: j3v0 | [Google drive](https://drive.google.com/file/d/1rD0QKEHdUSE-Cpijgxv4BlPUMRQ6Q69l/view?usp=sharing)]  
(8) SIP [[baidu pan](https://pan.baidu.com/s/1qvpfXrPYT94M6mD0pv3-SQ) fetch code: q0j5 | [Google drive](https://drive.google.com/file/d/1Ruv0oLVP8QjrN3keOtdCjSiX4mh7bBVN/view?usp=sharing)]  

## Acknowledgement
We thank the authors of [VST](https://github.com/yitu-opensource/T2T-ViT) for providing T2T-ViT backbone, the authors of [DPT](https://github.com/isl-org/DPT) for providing us the method to get estimated depth maps of single underwater images in USOD10K, the authors of [SVAM-Net](http://www.roboticsproceedings.org/rss18/p048.pdf) for providing the [USOD](https://irvlab.cs.umn.edu/resources/usod-dataset) dataset, and [Zhao Zhang](https://github.com/zzhanghub/eval-co-sod) for providing the efficient evaluation tool.

## Note to active participants
To spark research in the USOD research community, we discuss several potential use cases and applications of the USOD10K dataset and the USOD methods in the paper, and highlight some promising research directions for this young but challenging field.

**We hope our work will boost the development of USOD research. However, as a young research field, USOD is still far from being solved, leaving large room for further improvement** !!! 




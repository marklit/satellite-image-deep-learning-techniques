# 14. Super-resolution

<p align="center">
  <img src="images/super-res.jpg" width="500">
  <br>
  <b>Super resolution using multiple low resolution images as input.</b>
</p>

Super-resolution is a technique aimed at improving the resolution of an imaging system. This process can be applied prior to other image processing steps to increase the visibility of small objects or boundaries. Despite its potential benefits, the use of super-resolution is controversial due to the possibility of introducing artifacts that could be mistaken for real features. Super-resolution techniques are broadly categorized into two groups: single image super-resolution (SISR) and multi-image super-resolution (MISR). SISR focuses on enhancing the resolution of a single image, while MISR utilizes multiple images of the same scene to create a high-resolution output. Each approach has its own advantages and limitations, and the choice of method depends on the specific application and desired outcome. [Image source](https://github.com/worldstrat/worldstrat).

### 14.1. Multi image super-resolution (MISR)
Note that nearly all the MISR publications resulted from the [PROBA-V Super Resolution competition](https://kelvins.esa.int/proba-v-super-resolution/)

  14.1.1. [deepsum](https://github.com/diegovalsesia/deepsum) -> Deep neural network for Super-resolution of Unregistered Multitemporal images (ESA PROBA-V challenge)

  14.1.2. [3DWDSRNet](https://github.com/frandorr/3DWDSRNet) -> code to reproduce Satellite Image Multi-Frame Super Resolution (MISR) Using 3D Wide-Activation Neural Networks

  14.1.3. [RAMS](https://github.com/EscVM/RAMS) -> Official TensorFlow code for paper Multi-Image Super Resolution of Remotely Sensed Images Using Residual Attention Deep Neural Networks

  14.1.4. [TR-MISR](https://github.com/Suanmd/TR-MISR) ->  Transformer-based MISR framework for the the PROBA-V super-resolution challenge. With [paper](https://ieeexplore.ieee.org/abstract/document/9684717)

  14.1.5. [HighRes-net](https://github.com/ElementAI/HighRes-net) -> Pytorch implementation of HighRes-net, a neural network for multi-frame super-resolution, trained and tested on the European Space Agency’s Kelvin competition

  14.1.6. [ProbaVref](https://github.com/centreborelli/ProbaVref) -> Repurposing the Proba-V challenge for reference-aware super resolution

  14.1.7. [The missing ingredient in deep multi-temporal satellite image super-resolution](https://towardsdatascience.com/the-missing-ingredient-in-deep-multi-temporal-satellite-image-super-resolution-78cac0f063d9) -> Permutation invariance harnesses the power of ensembles in a single model, with repo [piunet](https://github.com/diegovalsesia/piunet)

  14.1.8. [MSTT-STVSR](https://github.com/XY-boy/MSTT-STVSR) -> Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer, JAG, 2022

  14.1.9. [Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites](https://centreborelli.github.io/HDR-DSP-SR/)

  14.1.10. [DDRN](https://github.com/kuijiang94/DDRN) -> Deep Distillation Recursive Network for Video Satellite Imagery Super-Resolution

  14.1.11. [worldstrat](https://github.com/worldstrat/worldstrat) -> SISR and MISR implementations of SRCNN

  14.1.12. [MISR-GRU](https://github.com/rarefin/MISR-GRU) -> Pytorch implementation of MISR-GRU, a deep neural network for multi image super-resolution (MISR), for ProbaV Super Resolution Competition

  14.1.13. [MSDTGP](https://github.com/XY-boy/MSDTGP) -> code for 2021 paper: Satellite Video Super-Resolution via Multiscale Deformable Convolution Alignment and Temporal Grouping Projection

  14.1.14. [proba-v-super-resolution-challenge](https://github.com/cedricoeldorf/proba-v-super-resolution-challenge) -> Solution to ESA's satellite imagery super resolution challenge

  14.1.15. [PROBA-V-Super-Resolution](https://github.com/spicy-mama/PROBA-V-Super-Resolution) -> solution using a custom deep learning architecture

  14.1.16. [satlas-super-resolution](https://github.com/allenai/satlas-super-resolution) -> Satlas Super Resolution: model is an adaptation of ESRGAN, with changes that allow the input to be a time series of Sentinel-2 images.

  14.1.17 [MISR Remote Sensing SRGAN](https://github.com/simon-donike/Remote-Sensing-SRGAN) -> PyTorch SRGAN for RGB Remote Sensing imagery, performing both SISR and MISR. MISR implementation inspired by RecursiveNet (HighResNet). Includes pretrained Checkpoints.

### 14.2. Single image super-resolution (SISR)

  14.2.1. [Super Resolution for Satellite Imagery - srcnn repo](https://github.com/WarrenGreen/srcnn)

  14.2.2. [TensorFlow implementation of "Accurate Image Super-Resolution Using Very Deep Convolutional Networks" adapted for working with geospatial data](https://github.com/CosmiQ/VDSR4Geo)

  14.2.3. [Random Forest Super-Resolution (RFSR repo)](https://github.com/jshermeyer/RFSR) including [sample data](https://github.com/jshermeyer/RFSR/tree/master/SampleImagery)

  14.2.4. [Enhancing Sentinel 2 images by combining Deep Image Prior and Decrappify](https://medium.com/omdena/pushing-the-limits-of-open-source-data-enhancing-satellite-imagery-through-deep-learning-9d8a3bbc0e0a). Repo for [deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior) and article on [decrappify](https://www.fast.ai/2019/05/03/decrappify/)

  14.2.5. [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/) -> the keras docs have a great tutorial on this light weight but well performing model

  14.2.6. [super-resolution-using-gan](https://github.com/saraivaufc/super-resolution-using-gan) -> Super-Resolution of Sentinel-2 Using Generative Adversarial Networks

  14.2.7. [Super-resolution of Multispectral Satellite Images Using Convolutional Neural Networks](https://up42.com/blog/tech/super-resolution-of-multispectral-satellite-images-using-convolutional) with [paper](https://arxiv.org/abs/2002.00580)

  14.2.8. [Multi-temporal Super-Resolution on Sentinel-2 Imagery](https://medium.com/sentinel-hub/multi-temporal-super-resolution-on-sentinel-2-imagery-6089c2b39ebc) using HighRes-Net, [repo](https://github.com/sentinel-hub/multi-temporal-super-resolution)

  14.2.9. [SSPSR-Pytorch](https://github.com/junjun-jiang/SSPSR) -> A spatial-spectral prior deep network for single hyperspectral image super-resolution

  14.2.10. [Sentinel-2 Super-Resolution: High Resolution For All (Bands)](https://up42.com/blog/tech/sentinel-2-superresolution)

  14.2.11. [CinCGAN](https://github.com/Junshk/CinCGAN-pytorch) -> Unofficial Implementation of [Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks](https://arxiv.org/abs/1809.00437)

  14.2.12. [Satellite-image-SRGAN using PyTorch](https://github.com/xjohnxjohn/Satellite-image-SRGAN)

  14.2.13. [EEGAN](https://github.com/kuijiang0802/EEGAN) -> Edge Enhanced GAN For Remote Sensing Image Super-Resolution, TensorFlow 1.1

  14.2.14. [PECNN](https://github.com/kuijiang0802/PECNN) -> A Progressively Enhanced Network for Video Satellite Imagery Super-Resolution, minimal documentation

  14.2.15. [hs-sr-tvtv](https://github.com/marijavella/hs-sr-tvtv) -> Enhanced Hyperspectral Image Super-Resolution via RGB Fusion and TV-TV Minimization

  14.2.16. [sr4rs](https://github.com/remicres/sr4rs) -> Super resolution for remote sensing, with pre-trained model for Sentinel-2, SRGAN-inspired

  14.2.17. [Restoring old aerial images with Deep Learning](https://towardsdatascience.com/restoring-old-aerial-images-with-deep-learning-60f0cfd2658) -> Medium article on Super Resolution with Perceptual Loss function and real images as input

  14.2.18. [RFSR_TGRS](https://github.com/wxywhu/RFSR_TGRS) -> code for the paper Hyperspectral Image Super-Resolution via Recurrent Feedback Embedding and Spatial-Spectral Consistency Regularization

  14.2.19. [SEN2VENµS](https://zenodo.org/record/6514159#.YoRxM5PMK3I) -> a dataset for the training of Sentinel-2 super-resolution algorithms. With [paper](https://www.mdpi.com/2306-5729/7/7/96)

  14.2.20. [TransENet](https://github.com/Shaosifan/TransENet) -> code for 2021 paper: Transformer-based Multi-Stage Enhancement for Remote Sensing Image Super-Resolution

  14.2.21. [SG-FBGAN](https://github.com/hanlinwu/SG-FBGAN) -> code for 2020 paper: Remote Sensing Image Super-Resolution via Saliency-Guided Feedback GANs

  14.2.22. [finetune_ESRGAN](https://github.com/johnjaniczek/finetune_ESRGAN) -> finetune the ESRGAN super resolution generator for remote sensing images and video

  14.2.23. [MIP](https://github.com/jiaming-wang/MIP) -> code for 2021 [paper](https://arxiv.org/abs/2105.03579): Unsupervised Remote Sensing Super-Resolution via Migration Image Prior

  14.2.24. [Optical-RemoteSensing-Image-Resolution](https://github.com/wenjiaXu/Optical-RemoteSensing-Image-Resolution) -> code for 2018 [paper](https://www.mdpi.com/2072-4292/10/12/1893): Deep Memory Connected Neural Network for Optical Remote Sensing Image Restoration. Two applications: Gaussian image denoising and single image super-resolution

  14.2.25. [HSENet](https://github.com/Shaosifan/HSENet) -> code for 2021 paper: Hybrid-Scale Self-Similarity Exploitation for Remote Sensing Image Super-Resolution

  14.2.26. [SR_RemoteSensing](https://github.com/Jing25/SR_RemoteSensing) -> Super-Resolution deep learning models for remote sensing data based on [BasicSR](https://github.com/XPixelGroup/BasicSR)

  14.2.27. [RSI-Net](https://github.com/EricBrock/RSI-Net) -> code for 2022 paper: A Deep Multi-task Convolutional Neural Network for Remote Sensing Image Super-resolution and Colorization

  14.2.28. [EDSR-Super-Resolution](https://github.com/RakeshRaj97/EDSR-Super-Resolution) -> EDSR model using PyTorch applied to satellite imagery

  14.2.29. [CycleCNN](https://github.com/haopzhang/CycleCNN) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9151194): Nonpairwise-Trained Cycle Convolutional Neural Network for Single Remote Sensing Image Super-Resolution

  14.2.30. [SISR with with Real-World Degradation Modeling](https://github.com/zhangjizhou-bit/Single-image-Super-Resolution-of-Remote-Sensing-Images-with-Real-World-Degradation-Modeling) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/12/2895): Single-Image Super Resolution of Remote Sensing Images with Real-World Degradation Modeling

  14.2.31. [pixel-smasher](https://github.com/ekcomputer/pixel-smasher) -> code for 2020 [paper](https://www.tandfonline.com/doi/abs/10.1080/07038992.2021.1924646?journalCode=ujrs20): Super-Resolution Surface Water Mapping on the Canadian Shield Using Planet CubeSat Images and a Generative Adversarial Network

  14.2.32. [satellite-image-super-resolution](https://github.com/farahmand-m/satellite-image-super-resolution) -> A Comparative Study on CNN-Based Single-Image Super-Resolution Techniques for Satellite Images

  14.2.33. [SatelliteSR](https://github.com/kmalhan/SatelliteSR) -> comparison of a number of techniques on the DOTA dataset

  14.2.34. [Image-Super-Resolution](https://github.com/Elangoraj/Image-Super-Resolution) -> Super resolution RESNET network

  14.2.35. [Unsupervised Super Resolution for Sentinel-2 satellite imagery](https://github.com/savassif/Thesis) -> using Deep Image Prior (DIP), Zero-Shot Super Resolution (ΖSSR) & Degradation-Aware Super Resolution (DASR)

  14.2.36. [Spectral Super-Resolution of Satellite Imagery with Generative Adversarial Networks](https://github.com/ImDanielRojas/thesis)

  14.2.37. [Super resolution using GAN / 4x Improvement](https://github.com/purijs/satellite-superresolution) -> applied to Sentinel 2

  14.2.38. [rs-esrgan](https://github.com/luissalgueiro/rs-esrgan) -> code for paper: RS-ESRGAN: Super-Resolution of Sentinel-2 Imagery Using Generative Adversarial Networks

  14.2.39. [TS-RSGAN](https://github.com/yicrane/TS-RSGAN) -> code for [paper](https://www.mdpi.com/2079-9292/11/21/3474): Super-Resolution of Remote Sensing Images for ×4 Resolution without Reference Images. Applied to Sentinel-2

  14.2.40. [CDCR](https://github.com/Suanmd/CDCR) -> code for 2023 paper: Combining Discrete and Continuous Representation: Scale-Arbitrary Super-Resolution for Satellite Images

  14.2.41. [FunSR](https://github.com/KyanChen/FunSR) -> code for 2023 paper: Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space

  14.2.42. [HAUNet_RSISR](https://github.com/likakakaka/HAUNet_RSISR) -> code for 2023 paper: Hybrid Attention-Based U-Shaped Network for Remote Sensing Image Super-Resolution

### 14.3. Super-resolution - Miscellaneous

  14.3.1. [The value of super resolution — real world use case](https://medium.com/sentinel-hub/the-value-of-super-resolution-real-world-use-case-2ba811f4cd7f) -> Medium article on parcel boundary detection with super-resolved satellite imagery

  14.3.2. [Super-Resolution on Satellite Imagery using Deep Learning](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-1-ec5c5cd3cd2) -> Nov 2016 blog post by CosmiQ Works with a nice introduction to the topic. Proposes and demonstrates a new architecture with perturbation layers with practical guidance on the methodology and [code](https://github.com/CosmiQ/super-resolution). [Three part series](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-3-2e2f61eee1d3)

  14.3.3. [Introduction to spatial resolution](https://medium.com/sentinel-hub/the-most-misunderstood-words-in-earth-observation-d0106adbe4b0)

  14.3.4. [Awesome-Super-Resolution](https://github.com/ptkin/Awesome-Super-Resolution) -> another 'awesome' repo, getting a little out of date now

  14.3.5. [Super-Resolution (python) Utilities for managing large satellite images](https://github.com/jshermeyer/SR_Utils)

  14.3.6. [pytorch-enhance](https://github.com/isaaccorley/pytorch-enhance) -> Library of Image Super-Resolution Models, Datasets, and Metrics for Benchmarking or Pretrained Use. Also [checkout this implementation in Jax](https://github.com/isaaccorley/jax-enhance)

  14.3.7. [Super Resolution in OpenCV](https://learnopencv.com/super-resolution-in-opencv/)

  14.3.8. [AI-based Super resolution and change detection to enforce Sentinel-2 systematic usage](https://medium.com/@sistema_gmbh/ai-based-super-resolution-and-change-detection-to-enforce-sentinel-2-systematic-usage-65aa37d0365) -> Worldview-2 images (2m) were used to create a reference dataset and increase the spatial resolution of the Copernicus sensor from 10m to 5m

  14.3.9. [SRCDNet](https://github.com/liumency/SRCDNet) -> The pytorch implementation for "Super-resolution-based Change Detection Network with Stacked Attention Module for Images with Different Resolutions ". SRCDNet is designed to learn and predict change maps from bi-temporal images with different resolutions

  14.3.10. [Model-Guided Deep Hyperspectral Image Super-resolution](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution) -> code accompanying the paper: Model-Guided Deep Hyperspectral Image Super-Resolution

  14.3.11. [Super-resolving beyond satellite hardware](https://github.com/smpetrie/superres) -> [paper](https://arxiv.org/abs/2103.06270) assessing SR performance in reconstructing realistically degraded satellite images

  14.3.12. [satellite-pixel-synthesis-pytorch](https://github.com/KellyYutongHe/satellite-pixel-synthesis-pytorch) -> PyTorch implementation of NeurIPS 2021 paper: Spatial-Temporal Super-Resolution of Satellite Imagery via Conditional Pixel Synthesis

  14.3.13. [SRE-HAN](https://github.com/bostankhan6/SRE-HAN) -> Squeeze-and-Residual-Excitation Holistic Attention Network improves super-resolution (SR) on remote-sensing imagery compared to other state-of-the-art attention-based SR models

  14.3.14. [satsr](https://github.com/deephdc/satsr) -> A project to perform super-resolution on multispectral images from any satellite, including Sentinel 2, Landsat 8, VIIRS &MODIS

  14.3.15. [OLI2MSI](https://github.com/wjwjww/OLI2MSI) -> dataset for remote sensing imagery super-resolution composed of Landsat8-OLI and Sentinel2-MSI images

  14.3.16. [MMSR](https://github.com/palmdong/MMSR) -> Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution

  14.3.17. [HSRnet](https://github.com/liangjiandeng/HSRnet) -> code for the 2021 [paper](https://arxiv.org/abs/2005.14400): Hyperspectral Image Super-resolution via Deep Spatio-spectral Attention Convolutional Neural Networks

  14.3.18. [RRSGAN](https://github.com/dongrunmin/RRSGAN) -> code for 2021 paper: RRSGAN: Reference-Based Super-Resolution for Remote Sensing Image

  14.3.19. [HDR-DSP-SR](https://github.com/centreborelli/HDR-DSP-SR) -> code for 2021 paper: Self-supervised multi-image super-resolution for push-frame satellite images

  14.3.20. [GAN-HSI-SR](https://github.com/ZhuangChen25674/GAN-HSI-SR) -> code for 2020 paper: Hyperspectral Image Super-Resolution by Band Attention Through Adversarial Learning

  14.3.21. [Restoring old aerial images with Deep Learning](https://towardsdatascience.com/restoring-old-aerial-images-with-deep-learning-60f0cfd2658) -> Medium article Super Resolution with Perceptual Loss function and real images as input
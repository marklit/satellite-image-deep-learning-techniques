# 7. Cloud detection & removal

<p align="center">
  <img src="images/clouds.png" width="550">
  <br>
  <b>(left) False colour image and (right) a cloud & shadow mask.</b>
</p>

Clouds are a major issue in remote sensing images as they can obscure the underlying ground features. This hinders the accuracy and effectiveness of remote sensing analysis, as the obscured regions cannot be properly interpreted. In order to address this challenge, various techniques have been developed to detect clouds in remote sensing images. Both classical algorithms and deep learning approaches can be employed for cloud detection. Classical algorithms typically use threshold-based techniques and hand-crafted features to identify cloud pixels. However, these techniques can be limited in their accuracy and are sensitive to changes in image appearance and cloud structure. On the other hand, deep learning approaches leverage the power of convolutional neural networks (CNNs) to accurately detect clouds in remote sensing images. These models are trained on large datasets of remote sensing images, allowing them to learn and generalize the unique features and patterns of clouds. The generated cloud mask can be used to identify the cloud pixels and eliminate them from further analysis or, alternatively, cloud inpainting techniques can be used to fill in the gaps left by the clouds. This approach helps to improve the accuracy of remote sensing analysis and provides a clearer view of the ground, even in the presence of clouds. Image adapted from [this source](https://www.sciencedirect.com/science/article/pii/S1877050922005361)

  7.1. [CloudSEN12](https://github.com/cloudsen12) -> Sentinel 2 cloud dataset with a [varierty of models here](https://github.com/cloudsen12/models)

  7.2. From [this article on sentinelhub](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13) there are three popular classical algorithms that detects thresholds in multiple bands in order to identify clouds. In the same article they propose using semantic segmentation combined with a CNN for a cloud classifier (excellent review paper [here](https://arxiv.org/pdf/1704.06857.pdf)), but state that this requires too much compute resources.

  7.3. [This article](https://www.mdpi.com/2072-4292/8/8/666) compares a number of ML algorithms, random forests, stochastic gradient descent, support vector machines, Bayesian method.

  7.4. [Segmentation of Clouds in Satellite Images Using Deep Learning](https://medium.com/swlh/segmentation-of-clouds-in-satellite-images-using-deep-learning-a9f56e0aa83d) -> semantic segmentation using a Unet on the Kaggle 38-Cloud dataset

  7.5. [Cloud Detection in Satellite Imagery](https://www.azavea.com/blog/2021/02/08/cloud-detection-in-satellite-imagery/) compares FPN+ResNet18 and CheapLab architectures on Sentinel-2 L1C and L2A imagery

  7.6. [Benchmarking Deep Learning models for Cloud Detection in Landsat-8 and Sentinel-2 images](https://github.com/IPL-UV/DL-L8S2-UV)

  7.7. [Landsat-8 to Proba-V Transfer Learning and Domain Adaptation for Cloud detection](https://github.com/IPL-UV/pvl8dagans)

  7.8. [Multitemporal Cloud Masking in Google Earth Engine](https://github.com/IPL-UV/ee_ipl_uv)

  7.9. [s2cloudmask](https://github.com/daleroberts/s2cloudmask) -> Sentinel-2 Cloud and Shadow Detection using Machine Learning

  7.10. [sentinel2-cloud-detector](https://github.com/sentinel-hub/sentinel2-cloud-detector) -> Sentinel Hub Cloud Detector for Sentinel-2 images in Python

  7.11. [dsen2-cr](https://github.com/ameraner/dsen2-cr) -> cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion, contains the model code, written in Python/Keras, as well as links to pre-trained checkpoints and the SEN12MS-CR dataset

  7.12. [pyatsa](https://github.com/agroimpacts/pyatsa) -> Python package implementing the Automated Time-Series Analysis method for masking clouds in satellite imagery developed by Zhu and Helmer 2018

  7.13. [decloud](https://github.com/CNES/decloud) -> Decloud enables the training of various deep nets to remove clouds in optical image, using e.g. Sentinel 1 & 2

  7.14. [cloudless](https://github.com/BradNeuberg/cloudless) -> Deep learning pipeline for orbital satellite data for detecting clouds

  7.15. [Deep-Gapfill](https://github.com/remicres/Deep-Gapfill) -> Official implementation of Optical image gap filling using deep convolutional autoencoder from optical and radar images

  7.16. [satellite-cloud-removal-dip](https://github.com/cidcom/satellite-cloud-removal-dip) -> Satellite cloud removal with Deep Image Prior, with [paper](https://www.mdpi.com/2072-4292/14/6/1342)

  7.17. [cloudFCN](https://github.com/aliFrancis/cloudFCN) -> Python 3 package for Fully Convolutional Network development, specifically for cloud masking

  7.18. [Fmask](https://github.com/GERSL/Fmask) -> Fmask (Function of mask) is used for automated clouds, cloud shadows, snow, and water masking for Landsats 4-9 and Sentinel 2 images, in Matlab. Also see [PyFmask](https://github.com/akalenda/PyFmask)

  7.19. [HOW TO USE DEEP LEARNING, PYTORCH LIGHTNING, AND THE PLANETARY COMPUTER TO PREDICT CLOUD COVER IN SATELLITE IMAGERY](https://www.drivendata.co/blog/cloud-cover-benchmark/)

  7.20. [cloud-cover-winners](https://github.com/drivendataorg/cloud-cover-winners) -> Code from the winning submissions for the On Cloud N: Cloud Cover Detection Challenge

  7.21. [On-Cloud-N: Cloud Cover Detection Challenge - 19th Place Solution](https://github.com/max-schaefer-dev/on-cloud-n-19th-place-solution)

  7.22. [ukis-csmask](https://github.com/dlr-eoc/ukis-csmask) -> package to masks clouds in Sentinel-2, Landsat-8, Landsat-7 and Landsat-5 images

  7.23. [OpenSICDR](https://github.com/dr-lizhiwei/OpenSICDR) -> long list of satellite image cloud detection resources

  7.24. [RS-Net](https://github.com/JacobJeppesen/RS-Net) -> code for the paper: A cloud detection algorithm for satellite imagery based on deep learning

  7.25. [Clouds-Segmentation-Project](https://github.com/TamirShalev/Clouds-Segmentation-Project) -> treats as a 3 class problem; Open clouds, Closed clouds and no clouds, uses pytorch on a dataset that consists of IR & Visual Grayscale images

  7.26. [STGAN](https://github.com/ermongroup/STGAN) -> PyTorch Implementation of STGAN for Cloud Removal in Satellite Images, with [paper](https://arxiv.org/abs/1912.06838)

  7.27. [mcgan-cvprw2017-pytorch](https://github.com/enomotokenji/mcgan-cvprw2017-pytorch) -> code for 2017 paper: Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets

  7.28. [Cloud-Net: A semantic segmentation CNN for cloud detection](https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection) -> an end-to-end cloud detection algorithm for Landsat 8 imagery, trained on 38-Cloud Training Set

  7.29. [fcd](https://github.com/jnyborg/fcd) -> code for 2021 paper: Fixed-Point GAN for Cloud Detection. A weakly-supervised approach, training with only image-level labels

  7.30. [CloudX-Net](https://github.com/sumitkanu/CloudX-Net) -> an efficient and robust architecture used for detection of clouds from satellite images

  7.31. [A simple cloud-detection walk-through using Convolutional Neural Network (CNN and U-Net) and fast.ai library](https://medium.com/analytics-vidhya/a-simple-cloud-detection-walk-through-using-convolutional-neural-network-cnn-and-u-net-and-bc745dda4b04)

  7.32. [38Cloud-Medium](https://github.com/cordmaur/38Cloud-Medium) -> Walk-through using u-net to detect clouds in satellite images with fast.ai

  7.33. [cloud_detection_using_satellite_data](https://github.com/ZhouPeng-NIMST/cloud_detection_using_satellite_data) -> performed on Sentinel 2 data

  7.34. [Luojia1-Cloud-Detection](https://github.com/dedztbh/Luojia1-Cloud-Detection) -> Luojia-1 Satellite Visible Band Nighttime Imagery Cloud Detection

  7.35. [SEN12MS-CR-TS](https://github.com/PatrickTUM/SEN12MS-CR-TS) -> code for 2022 paper: A Remote Sensing Data Set for Multi-modal Multi-temporal Cloud Removal

  7.36. [ES-CCGAN](https://github.com/AnnaCUG/ES-CCGAN) -> This is a dehazed method for remote sensing image, which based on CycleGAN

  7.37. [Cloud_Classification_DL](https://github.com/nishp763/Cloud_Classification_DL) -> Classifying cloud organization patterns from satellite images using Deep Learning techniques (Mask R-CNN)

  7.38. [CNN-based-Cloud-Detection-Methods](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods) -> Understanding the Role of Receptive Field of Convolutional Neural Network for Cloud Detection in Landsat 8 OLI Imagery

  7.39. [cloud-removal-deploy](https://github.com/XavierJiezou/cloud-removal-deploy) -> flask app for cloud removal

  7.40. [CloudMattingGAN](https://github.com/flyakon/CloudMattingGAN) -> code for 2019 paper: Generative Adversarial Training for Weakly Supervised Cloud Matting

  7.41. [atrain-cloudseg](https://github.com/seanremy/atrain-cloudseg) -> Official repository for the A-Train Cloud Segmentation Dataset

  7.42. [CDnet](https://github.com/nkszjx/CDnet-pytorch-master) -> code for 2019 paper: CNN-Based Cloud Detection for Remote Sensing Imager

  7.43. [GLNET](https://github.com/wuchangsheng951/GLNET) -> code for 2021 paper: Convolutional Neural Networks Based Remote Sensing Scene Classification under Clear and Cloudy Environments

  7.44. [CDnetV2](https://github.com/nkszjx/CDnetV2-pytorch-master) -> code for 2021 paper: CNN-Based Cloud Detection for Remote Sensing Imagery With Cloud-Snow Coexistence

  7.45. [grouped-features-alignment](https://github.com/nkszjx/grouped-features-alignment) -> code for 2021 paper: Unsupervised Domain Adaptation for Cloud Detection Based on Grouped Features Alignment and Entropy Minimization

  7.46. [Detecting Cloud Cover Via Sentinel-2 Satellite Data](https://benjaminwarner.dev/2022/03/11/detecting-cloud-cover-via-satellite) -> blog post on Benjamin Warners Top-10 Percent Solution to DrivenData’s On CloudN Competition using fast.ai & customized version of XResNeXt50. [Repo](https://github.com/warner-benjamin/code_for_blog_posts/tree/main/2022/drivendata_cloudn)

  7.47. [AISD](https://github.com/RSrscoder/AISD) -> code (Matlab) and dataset for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271620302045): Deeply supervised convolutional neural network for shadow detection based on a novel aerial shadow imagery dataset

  7.48. [CloudGAN](https://github.com/JerrySchonenberg/CloudGAN) -> Detecting and Removing Clouds from RGB-images using Image Inpainting

  7.49. [Using GANs to Augment Data for Cloud Image Segmentation Task](https://github.com/jain15mayank/GAN-augmentation-cloud-image-segmentation) -> code for 2021 [paper](https://arxiv.org/abs/2106.03064)

  7.50. [Cloud-Segmentation-from-Satellite-Imagery](https://github.com/vedantk-b/Cloud-Segmentation-from-Satellite-Imagery) -> applied to Sentinel-2 dataset

  7.51. [HRC_WHU](https://github.com/dr-lizhiwei/HRC_WHU) -> High-Resolution Cloud Detection Dataset comprising 150 RGB images and a resolution varying from 0.5 to 15 m in different global regions

  7.52. [MEcGANs](https://github.com/andrzejmizera/MEcGANs) -> Cloud Removal from Satellite Imagery using Multispectral Edge-filtered Conditional Generative Adversarial Networks

  7.53. [CloudXNet](https://github.com/shyamfec/CloudXNet) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S2352938520303803): CloudX-net: A robust encoder-decoder architecture for cloud detection from satellite remote sensing images

  7.54. [refined-unet-lite](https://github.com/92xianshen/refined-unet-lite) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S1877050922005361): Refined UNet Lite: End-to-End Lightweight Network for Edge-precise Cloud Detection

  7.55. [cloud-buster](https://github.com/azavea/cloud-buster) -> Sentinel-2 L1C and L2A Imagery with Fewer Clouds

  7.56. [SatelliteCloudGenerator](https://github.com/cidcom/SatelliteCloudGenerator) -> A PyTorch-based tool to generate clouds for satellite images

  7.57. [SEnSeI](https://github.com/aliFrancis/SEnSeI) -> A python 3 package for developing sensor independent deep learning models for cloud masking in satellite imagery

  7.58. [cloud-detection-venus](https://github.com/pesekon2/cloud-detection-venus) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/20/5210): Using Convolutional Neural Networks for Cloud Detection on VENμS Images over Multiple Land-Cover Types

  7.59. [explaining_cloud_effects](https://github.com/JakobCode/explaining_cloud_effects) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9956865): Explaining the Effects of Clouds on Remote Sensing Scene Classification

  7.60. [Clouds-Images-Segmentation](https://github.com/DavidHuji/Clouds-Images-Segmentation) -> Marine Stratocumulus Cloud-Type Classification from SEVIRI Using Convolutional Neural Networks

  7.61. [DeCloud-GAN](https://github.com/pixiedust18/DeCloud-GAN) -> code for 2021 paper: DeCloud GAN: An Advanced Generative Adversarial Network for Removing Cloud Cover in Optical Remote Sensing Imagery

  7.62. [km_predict](https://github.com/kappazeta/km_predict) -> KappaMask, or km-predict, is a cloud detector for Sentinel-2 Level-1C and Level-2A input products applied to S2 full image prediction

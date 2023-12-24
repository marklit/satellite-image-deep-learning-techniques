# 8. Change detection

<p align="center">
  <img src="images/change.png" width="950">
  <br>
  <b>(left) Initial and (middle) after some development, with (right) the change highlighted.</b>
</p>

Change detection is a vital component of remote sensing analysis, enabling the monitoring of landscape changes over time. This technique can be applied to identify a wide range of changes, including land use changes, urban development, coastal erosion, and deforestation. Change detection can be performed on a pair of images taken at different times, or by analyzing multiple images collected over a period of time. It is important to note that while change detection is primarily used to detect changes in the landscape, it can also be influenced by the presence of clouds and shadows. These dynamic elements can alter the appearance of the image, leading to false positives in change detection results. Therefore, it is essential to consider the impact of clouds and shadows on change detection analysis, and to employ appropriate methods to mitigate their influence. [Image source](https://www.mdpi.com/2072-4292/11/3/240)

  8.1. [awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) lists many datasets and publications

  8.2. [Change-Detection-Review](https://github.com/MinZHANG-WHU/Change-Detection-Review) -> A review of change detection methods, including code and open data sets for deep learning

  8.3. [Change Detection using Siamese Networks](https://towardsdatascience.com/change-detection-using-siamese-networks-fc2935fff82) -> Medium article `BEGINNER`

  8.4. [STANet](https://github.com/justchenhao/STANet) -> official implementation of the spatial-temporal attention neural network (STANet) for remote sensing image change detection `BEGINNER`

  8.5. [UNet-based-Unsupervised-Change-Detection](https://github.com/annabosman/UNet-based-Unsupervised-Change-Detection) -> A convolutional neural network (CNN) and semantic segmentation is implemented to detect the changes between the images, as well as classify the changes into the correct semantic class, with [arxiv paper](https://arxiv.org/abs/1812.05815) `BEGINNER`

  8.6. [BIT_CD](https://github.com/justchenhao/BIT_CD) -> Official Pytorch Implementation of Remote Sensing Image Change Detection with Transformers

  8.7. [Unstructured-change-detection-using-CNN](https://github.com/vbhavank/Unstructured-change-detection-using-CNN)

  8.8. [Siamese neural network to detect changes in aerial images](https://github.com/vbhavank/Siamese-neural-network-for-change-detection) -> uses Keras and VGG16 architecture

  8.9. [Change Detection in 3D: Generating Digital Elevation Models from Dove Imagery](https://www.planet.com/pulse/publications/change-detection-in-3d-generating-digital-elevation-models-from-dove-imagery/)

  8.10. [QGIS plugin for applying change detection algorithms on high resolution satellite imagery](https://github.com/dymaxionlabs/massive-change-detection)

  8.11. [LamboiseNet](https://github.com/hbaudhuin/LamboiseNet) -> Master thesis about change detection in satellite imagery using Deep Learning

  8.12. [Fully Convolutional Siamese Networks for Change Detection](https://github.com/rcdaudt/fully_convolutional_change_detection) -> with [paper](https://ieeexplore.ieee.org/abstract/document/8451652)

  8.13. [Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks](https://github.com/rcdaudt/patch_based_change_detection) -> with [paper](https://ieeexplore.ieee.org/abstract/document/8518015), used the Onera Satellite Change Detection (OSCD) dataset

  8.14. [IAug_CDNet](https://github.com/justchenhao/IAug_CDNet) -> Official Pytorch Implementation of Adversarial Instance Augmentation for Building Change Detection in Remote Sensing Images

  8.15. [dpm-rnn-public](https://github.com/olliestephenson/dpm-rnn-public) -> Code implementing a damage mapping method combining satellite data with deep learning

  8.16. [SenseEarth2020-ChangeDetection](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection) -> 1st place solution to the Satellite Image Change Detection Challenge hosted by SenseTime; predictions of five HRNet-based segmentation models are ensembled, serving as pseudo labels of unchanged areas

  8.17. [KPCAMNet](https://github.com/I-Hope-Peace/KPCAMNet) -> Python implementation of the paper Unsupervised Change Detection in Multi-temporal VHR Images Based on Deep Kernel PCA Convolutional Mapping Network

  8.18. [CDLab](https://github.com/Bobholamovic/CDLab) -> benchmarking deep learning-based change detection methods.

  8.19. [Siam-NestedUNet](https://github.com/likyoo/Siam-NestedUNet) -> The pytorch implementation for "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images"

  8.20. [SUNet-change_detection](https://github.com/ShaoRuizhe/SUNet-change_detection) -> Implementation of paper SUNet: Change Detection for Heterogeneous Remote Sensing Images from Satellite and UAV Using a Dual-Channel Fully Convolution Network

  8.21. [Self-supervised Change Detection in Multi-view Remote Sensing Images](https://github.com/cyx669521/self-supervised_change_detetction)

  8.22. [MFPNet](https://github.com/wzjialang/MFPNet) -> Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity

  8.23. [GitHub for the DIUx xView Detection Challenge](https://github.com/DIUx-xView) -> The xView2 Challenge focuses on automating the process of assessing building damage after a natural disaster

  8.24. [DASNet](https://github.com/lehaifeng/DASNet) -> Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images

  8.25. [Self-Attention for Raw Optical Satellite Time Series Classification](https://github.com/MarcCoru/crop-type-mapping)

  8.26. [planet-movement](https://github.com/rhammell/planet-movement) -> Find and process Planet image pairs to highlight object movement

  8.27. [temporal-cluster-matching](https://github.com/microsoft/temporal-cluster-matching) -> detecting change in structure footprints from time series of remotely sensed imagery

  8.28. [autoRIFT](https://github.com/nasa-jpl/autoRIFT) -> fast and intelligent algorithm for finding the pixel displacement between two images

  8.29. [DSAMNet](https://github.com/liumency/DSAMNet) -> Code for “A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection”. The main types of changes in the dataset include: (a) newly built urban buildings; (b) suburban dilation; (c) groundwork before construction; (d) change of vegetation; (e) road expansion; (f) sea construction.

  8.30. [SRCDNet](https://github.com/liumency/SRCDNet) -> The pytorch implementation for "Super-resolution-based Change Detection Network with Stacked Attention Module for Images with Different Resolutions ". SRCDNet is designed to learn and predict change maps from bi-temporal images with different resolutions

  8.31. [Land-Cover-Analysis](https://github.com/Kalit31/Land-Cover-Analysis) -> Land Cover Change Detection using Satellite Image Segmentation

  8.32. [A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images)

  8.33. [Satellite-Image-Alignment-Differencing-and-Segmentation](https://github.com/rishi5kesh/Satellite-Image-Alignment-Differencing-and-Segmentation) -> thesis on change detection

  8.34. [Change Detection in Multi-temporal Satellite Images](https://github.com/IhebeddineRyahi/Change-detection-in-multitemporal-satellite-images) -> uses Principal Component Analysis (PCA) and K-means clustering

  8.35. [Unsupervised Change Detection Algorithm using PCA and K-Means Clustering](https://github.com/leduckhai/Change-Detection-PCA-KMeans) -> in Matlab but has paper

  8.36. [ChangeFormer](https://github.com/wgcban/ChangeFormer) -> A Transformer-Based Siamese Network for Change Detection. Uses transformer architecture to address the limitations of CNN in handling multi-scale long-range details. Demonstrates that ChangeFormer captures much finer details compared to the other SOTA methods, achieving better performance on benchmark datasets

  8.37. [Heterogeneous_CD](https://github.com/llu025/Heterogeneous_CD) -> Heterogeneous Change Detection in Remote Sensing Images. Accompanies [Code-Aligned Autoencoders for Unsupervised Change Detection in Multimodal Remote Sensing Images](https://arxiv.org/abs/2004.07011)

  8.38. [ChangeDetectionProject](https://github.com/previtus/ChangeDetectionProject) -> Trying out Active Learning in with deep CNNs for Change detection on remote sensing data

  8.39. [DSFANet](https://github.com/rulixiang/DSFANet) -> Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images

  8.40. [siamese-change-detection](https://github.com/mvkolos/siamese-change-detection) -> Targeted synthesis of multi-temporal remote sensing images for change detection using siamese neural networks

  8.41. [Bi-SRNet](https://github.com/ggsDing/Bi-SRNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9721305): Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images

  8.42. [SiROC](https://github.com/lukaskondmann/SiROC) -> Implementation of the paper: Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images. Applied to Sentinel-2 and high-resolution Planetscope imagery on four datasets

  8.43. [DSMSCN](https://github.com/I-Hope-Peace/DSMSCN) -> Tensorflow implementation for Change Detection in Multi-temporal VHR Images Based on Deep Siamese Multi-scale Convolutional Neural Networks

  8.44. [RaVAEn](https://github.com/spaceml-org/RaVAEn) -> a lightweight, unsupervised approach for change detection in satellite data based on Variational Auto-Encoders (VAEs) with the specific purpose of on-board deployment. It flags changed areas to prioritise for downlink, shortening the response time

  8.45. [SemiCD](https://github.com/wgcban/SemiCD) -> Code for [paper](https://arxiv.org/abs/2204.08454): Revisiting Consistency Regularization for Semi-supervised Change Detection in Remote Sensing Images. Achieves the performance of supervised CD even with access to as little as 10% of the annotated training data

  8.46. [FCCDN_pytorch](https://github.com/chenpan0615/FCCDN_pytorch) -> code for [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000636): FCCDN: Feature Constraint Network for VHR Image Change Detection. Uses the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) building change detection dataset

  8.47. [INLPG_Python](https://github.com/zcsisiyao/INLPG_Python) -> code for paper: Structure Consistency based Graph for Unsupervised Change Detection with Homogeneous and Heterogeneous Remote Sensing Images

  8.48. [NSPG_Python](https://github.com/zcsisiyao/NSPG_Python) -> code for paper: Nonlocal patch similarity based heterogeneous remote sensing change detection

  8.49. [LGPNet-BCD](https://github.com/TongfeiLiu/LGPNet-BCD) -> code for 2021 paper: Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy

  8.50. [DS_UNet](https://github.com/SebastianHafner/DS_UNet) -> code for 2021 paper: Sentinel-1 and Sentinel-2 Data Fusion for Urban Change Detection using a Dual Stream U-Net, uses Onera Satellite Change Detection dataset

  8.51. [SiameseSSL](https://github.com/SebastianHafner/SiameseSSL) -> code for 2022 [paper](https://arxiv.org/abs/2204.12202): Urban change detection with a Dual-Task Siamese network and semi-supervised learning. Uses SpaceNet 7 dataset

  8.52. [CD-SOTA-methods](https://github.com/wgcban/CD-SOTA-methods) -> Remote sensing change detection: State-of-the-art methods and available datasets

  8.53. [multimodalCD_ISPRS21](https://github.com/PatrickTUM/multimodalCD_ISPRS21) -> code for 2021 paper: Fusing Multi-modal Data for Supervised Change Detection

  8.54. [Unsupervised-CD-in-SITS-using-DL-and-Graphs](https://github.com/ekalinicheva/Unsupervised-CD-in-SITS-using-DL-and-Graphs) -> code for article: Unsupervised Change Detection Analysis in Satellite Image Time Series using Deep Learning Combined with Graph-Based Approaches

  8.55. [LSNet](https://github.com/qaz670756/LSNet) -> code for 2022 [paper](https://arxiv.org/abs/2201.09156): Extremely Light-Weight Siamese Network For Change Detection in Remote Sensing Image

  8.56. [Change-Detection-in-Remote-Sensing-Images](https://github.com/themrityunjay/Change-Detection-in-Remote-Sensing-Images) ->  using PCA & K-means

  8.57. [End-to-end-CD-for-VHR-satellite-image](https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image) -> code for 2019 [paper](https://www.mdpi.com/2072-4292/11/11/1382): End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++

  8.58. [Semantic-Change-Detection](https://github.com/daifeng2016/Semantic-Change-Detection) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/pii/S0303243421001720): SCDNET: A novel convolutional network for semantic change detection in high resolution optical remote sensing imagery

  8.59. [ERCNN-DRS_urban_change_monitoring](https://github.com/It4innovations/ERCNN-DRS_urban_change_monitoring) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/15/3000): Neural Network-Based Urban Change Monitoring with Deep-Temporal Multispectral and SAR Remote Sensing Data

  8.60. [EGRCNN](https://github.com/luting-hnu/EGRCNN) -> code for 2021 paper: Edge-guided Recurrent Convolutional Neural Network for Multi-temporal Remote Sensing Image Building Change Detection

  8.61. [Unsupervised-Remote-Sensing-Change-Detection](https://github.com/TangXu-Group/Unsupervised-Remote-Sensing-Change-Detection) -> code for 2021 paper: An Unsupervised Remote Sensing Change Detection Method Based on Multiscale Graph Convolutional Network and Metric Learning

  8.62. [CropLand-CD](https://github.com/liumency/CropLand-CD) -> code for 2022 paper: A CNN-transformer Network with Multi-scale Context Aggregation for Fine-grained Cropland Change Detection

  8.63. [contrastive-surface-image-pretraining](https://github.com/isaaccorley/contrastive-surface-image-pretraining) -> code for 2022 [paper](https://arxiv.org/abs/2202.13251): Supervising Remote Sensing Change Detection Models with 3D Surface Semantics

  8.64. [dcvaVHROptical](https://github.com/sudipansaha/dcvaVHROptical) -> Deep Change Vector Analysis (DCVA) change detection. Code for 2019 paper: Unsupervised Deep Change Vector Analysis for Multiple-Change Detection in VHR Images

  8.65. [hyperdimensionalCD](https://github.com/sudipansaha/hyperdimensionalCD) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9582825): Change Detection in Hyperdimensional Images Using Untrained Models

  8.66. [DSFANet](https://github.com/wwdAlger/DSFANet) -> code for 2018 [paper](https://arxiv.org/abs/1812.00645): Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images

  8.67. [FCD-GAN-pytorch](https://github.com/Cwuwhu/FCD-GAN-pytorch) -> Fully Convolutional Change Detection Framework with Generative Adversarial Network (FCD-GAN) is a framework for change detection in multi-temporal remote sensing images

  8.68. [DARNet-CD](https://github.com/jimmyli08/DARNet-CD) -> code for 2022 paper: A Densely Attentive Refinement Network for Change Detection Based on Very-High-Resolution Bitemporal Remote Sensing Images

  8.69. [xView2_Vulcan](https://github.com/RitwikGupta/xView2-Vulcan) -> Damage assessment using pre and post orthoimagery. Modified + productionized model based off the first-place model from the xView2 challenge.

  8.70. [ESCNet](https://github.com/Bobholamovic/ESCNet) -> code for 2021 paper: An End-to-End Superpixel-Enhanced Change Detection Network for Very-High-Resolution Remote Sensing Images

  8.71. [ForestCoverChange](https://github.com/annusgit/ForestCoverChange) -> Detecting and Predicting Forest Cover Change in Pakistani Areas Using Remote Sensing Imagery

  8.72. [deforestation-detection](https://github.com/vldkhramtsov/deforestation-detection) -> code for 2020 paper: DEEP LEARNING FOR HIGH-FREQUENCY CHANGE DETECTION IN UKRAINIAN FOREST ECOSYSTEM WITH SENTINEL-2

  8.73. [forest_change_detection](https://github.com/QuantuMobileSoftware/forest_change_detection) -> forest change segmentation with time-dependent models, including Siamese, UNet-LSTM, UNet-diff, UNet3D models. Code for 2021 paper: Deep Learning for Regular Change Detection in Ukrainian Forest Ecosystem With Sentinel-2

  8.74. [SentinelClearcutDetection](https://github.com/vldkhramtsov/SentinelClearcutDetection) -> Scripts for deforestation detection on the Sentinel-2 Level-A images

  8.75. [clearcut_detection](https://github.com/QuantuMobileSoftware/clearcut_detection) -> research & web-service for clearcut detection

  8.76. [CDRL](https://github.com/cjf8899/CDRL) -> code for 2022 [paper](https://arxiv.org/abs/2204.01200): Unsupervised Change Detection Based on Image Reconstruction Loss

  8.77. [ddpm-cd](https://github.com/wgcban/ddpm-cd) -> code for 2022 [paper](https://arxiv.org/abs/2206.11892): Remote Sensing Change Detection (Segmentation) using Denoising Diffusion Probabilistic Models

  8.78. [Remote-sensing-time-series-change-detection](https://github.com/liulianni1688/Remote-sensing-time-series-change-detection) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0034425722001079): Graph-based block-level urban change detection using Sentinel-2 time series

  8.79. [austin-ml-change-detection-demo](https://github.com/makepath/austin-ml-change-detection-demo) -> A change detection demo for the Austin area using a pre-trained PyTorch model scaled with Dask on Planet imagery

  8.80. [dfc2021-msd-baseline](https://github.com/calebrob6/dfc2021-msd-baseline) -> A baseline for the "Multitemporal Semantic Change Detection" track of the 2021 IEEE GRSS Data Fusion Competition

  8.81. [CorrFusionNet](https://github.com/rulixiang/CorrFusionNet) -> code for 2020 [paper](https://arxiv.org/abs/2006.02176): Multi-Temporal Scene Classification and Scene Change Detection with Correlation based Fusion

  8.82. [ChangeDetectionPCAKmeans](https://github.com/rulixiang/ChangeDetectionPCAKmeans) -> MATLAB implementation for Unsupervised Change Detection in Satellite Images Using Principal Component Analysis and k-Means Clustering.

  8.83. [IRCNN](https://github.com/thebinyang/IRCNN) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9721897): IRCNN: An Irregular-Time-Distanced Recurrent Convolutional Neural Network for Change Detection in Satellite Time Series

  8.84. [UTRNet](https://github.com/thebinyang/UTRNet) -> An Unsupervised Time-Distance-Guided Convolutional Recurrent Network for Change Detection in Irregularly Collected Images

  8.85. [open-cd](https://github.com/likyoo/open-cd) -> an open source change detection toolbox based on a series of open source general vision task tools

  8.86. [Tiny_model_4_CD](https://github.com/AndreaCodegoni/Tiny_model_4_CD) -> code for 2022 [paper](https://arxiv.org/abs/2207.13159): TINYCD: A (Not So) Deep Learning Model For Change Detection. Uses LEVIR-CD & WHU-CD datasets

  8.87. [FHD](https://github.com/ZSVOS/FHD) -> code for 2022 paper: Feature Hierarchical Differentiation for Remote Sensing Image Change Detection

  8.88. [Change detection with Raster Vision](https://www.azavea.com/blog/2022/04/18/change-detection-with-raster-vision/) -> blog post with Colab notebook

  8.89. [building-expansion](https://github.com/reglab/building_expansion) -> code for 2021 [paper](https://arxiv.org/abs/2105.14159): Enhancing Environmental Enforcement with Near Real-Time Monitoring: Likelihood-Based Detection of Structural Expansion of Intensive Livestock Farms

  8.90. [SaDL_CD](https://github.com/justchenhao/SaDL_CD) -> code for 2022 [paper](https://arxiv.org/abs/2205.13769): Semantic-aware Dense Representation Learning for Remote Sensing Image Change Detection

  8.91. [EGCTNet_pytorch](https://github.com/chen11221/EGCTNet_pytorch) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/18/4524): Building Change Detection Based on an Edge-Guided Convolutional Neural Network Combined with a Transformer

  8.92. [S2-cGAN](https://git.tu-berlin.de/rsim/S2-cGAN) -> code for 2020 [paper](https://arxiv.org/abs/2007.02565): S2-cGAN: Self-Supervised Adversarial Representation Learning for Binary Change Detection in Multispectral Images

  8.93. [A-loss-function-for-change-detection](https://github.com/Chuan-shanjia/A-loss-function-for-change-detection) -> code for 2022 paper: UAL: Unchanged Area Loss-Function for Change Detection Networks

  8.94. [IEEE_TGRS_SSTFormer](https://github.com/yanhengwang-heu/IEEE_TGRS_SSTFormer) -> code for 2022 paper: Spectral–Spatial–Temporal Transformers for Hyperspectral Image Change Detection

  8.95. [DMINet](https://github.com/ZhengJianwei2/DMINet) -> code for 2023 paper: Change Detection on Remote Sensing Images Using Dual-Branch Multilevel Intertemporal Network

  8.96. [AFCF3D-Net](https://github.com/wm-Githuber/AFCF3D-Net) -> code for 2023 paper: Adjacent-level Feature Cross-Fusion with 3D CNN for Remote Sensing Image Change Detection

  8.97. [DSAHRNet](https://github.com/Githubwujinming/DSAHRNet) -> code for paper: A Deeply Attentive High-Resolution Network for Change Detection in Remote Sensing Images

  8.98. [RDPNet](https://github.com/Chnja/RDPNet) -> code for 2022 paper: RDP-Net: Region Detail Preserving Network for Change Detection

  8.99. [BGAAE_CD](https://github.com/xauter/BGAAE_CD) -> code for 2022 paper: Bipartite Graph Attention Autoencoders for Unsupervised Change Detection Using VHR Remote Sensing Images

  8.100. [Unsupervised-Change-Detection](https://github.com/voodooed/Unsupervised-Change-Detection) -> code for 2009 paper: Unsupervised Change Detection in Satellite Images Using Principal Component Analysis and k-Means Clustering

  8.101. [Metric-CD](https://github.com/wgcban/Metric-CD) -> code for 2023 paper: Deep Metric Learning for Unsupervised Change Detection in Remote Sensing Images

  8.102. [HANet-CD](https://github.com/ChengxiHAN/HANet-CD) -> code for 2023 paper: HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images

  8.103. [SRGCAE](https://github.com/ChenHongruixuan/SRGCAE) -> code for 2022 paper: Unsupervised Multimodal Change Detection Based on Structural Relationship Graph Representation Learning

  8.104. [change_detection_onera_baselines](https://github.com/previtus/change_detection_onera_baselines) -> Siamese version of U-Net baseline model

  8.105. [SiamCRNN](https://github.com/ChenHongruixuan/SiamCRNN) -> code for 2020 paper: Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network

  8.106. [Graph-based methods for change detection in remote sensing images](https://github.com/jfflorez/Graph-based-methods-for-change-detection-in-remote-sensing-images) -> code for paper: Graph Learning Based on Signal Smoothness Representation for Homogeneous and Heterogeneous Change Detection

  8.107. [TransUNetplus2](https://github.com/aj1365/TransUNetplus2) -> code for 2023 paper: TransU-Net++: Rethinking attention gated TransU-Net for deforestation mapping. Uses the Amazon and Atlantic forest dataset

  8.108. [AR-CDNet](https://github.com/guanyuezhen/AR-CDNet) -> code for 2023 paper: Towards Accurate and Reliable Change Detection of Remote Sensing Images via Knowledge Review and Online Uncertainty Estimation

  8.109. [CICNet](https://github.com/ZhengJianwei2/CICNet) -> code for 2023 paper: Compact Intertemporal Coupling Network for Remote Sensing Change Detection

  8.110. [BGINet](https://github.com/JackLiu-97/BGINet) -> code for 2023 [paper](https://arxiv.org/abs/2307.02007): Remote Sensing Image Change Detection with Graph Interaction

  8.111. [DSNUNet](https://github.com/NightSongs/DSNUNet) -> code for 2022 paper: DSNUNet: An Improved Forest Change Detection Network by Combining Sentinel-1 and Sentinel-2 Images

  8.112. [Forest-CD](https://github.com/NightSongs/Forest-CD) -> code for 2022 paper: Forest-CD: Forest Change Detection Network Based on VHR Images

  8.113. [S3Net_CD](https://github.com/OMEGA-RS/S3Net_CD) -> code for 2023 paper: Superpixel-Guided Self-Supervised Learning Network for Change Detection in Multitemporal Image Change Detection

  8.114. [T-UNet](https://github.com/Pl-2000/T-UNet) -> code for 2023 paper: T-UNet: Triplet UNet for Change Detection in High-Resolution Remote Sensing Images

  8.115. [UCDFormer](https://github.com/zhu-xlab/UCDFormer) -> code for 2023 paper: UCDFormer: Unsupervised Change Detection Using a Transformer-driven Image Translation

  8.116. [satellite-change-events](https://github.com/utkarshmall13/satellite-change-events) -> code for paper: Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery, uses Sentinel 2 CaiRoad & CalFire datasets

  8.117. [CACo](https://github.com/utkarshmall13/CACo) -> code for paper: Change-Aware Sampling and Contrastive Learning for Satellite Images

  8.118. [LightCDNet](https://github.com/NightSongs/LightCDNet) -> code for 2023 paper: LightCDNet: Lightweight Change Detection Network Based on VHR Images

  8.119. [OpenMineChangeDetection](https://github.com/Dibz15/OpenMineChangeDetection) -> code for thesis: Characterising Open Cast Mining from Satellite Data (Sentinel 2), implements TinyCD, LSNet & DDPM-CD

  8.120. [multi-task-L-UNet](https://github.com/mpapadomanolaki/multi-task-L-UNet) -> code for 2021 paper: A Deep Multi-Task Learning Framework Coupling Semantic Segmentation and Fully Convolutional LSTM Networks for Urban Change Detection. Applied to SpaceNet7 dataset

  8.121. [urban_change_detection](https://github.com/SebastianHafner/urban_change_detection) -> code for 2019 paper: Detecting Urban Changes With Recurrent Neural Networks From Multitemporal Sentinel-2 Data. [fabric](https://github.com/granularai/fabric) is another implementation

  8.122. [UNetLSTM](https://github.com/mpapadomanolaki/UNetLSTM) -> code for 2019 paper: Detecting Urban Changes With Recurrent Neural Networks From Multitemporal Sentinel-2 Data

  8.123. [SDACD](https://github.com/Perfect-You/SDACD) -> An End-to-end Supervised Domain Adaptation Framework for Cross-domain Change Detection

  8.124. [CycleGAN-Based-DA-for-CD](https://github.com/pjsoto/CycleGAN-Based-DA-for-CD) -> CycleGAN-based Domain Adaptation for Deforestation Detection

  8.125. [CGNet-CD](https://github.com/ChengxiHAN/CGNet-CD) -> code for 2023 paper: Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery

  8.126. [PA-Former](https://github.com/liumency/PA-Former) -> code for 2022 paper: PA-Former: Learning Prior-Aware Transformer for Remote Sensing Building Change Detection

  8.127. [AERNet](https://github.com/zjd1836/AERNet) -> code for 2023 paper: AERNet: An Attention-Guided Edge Refinement Network and a Dataset for Remote Sensing Building Change Detection (HRCUS-CD)

  8.128. [S1GFlood-Detection](https://github.com/Tamer-Saleh/S1GFlood-Detection) -> code for 2023 paper: DAM-Net: Global Flood Detection from SAR Imagery Using Differential Attention Metric-Based Vision Transformers. Includes S1GFloods dataset

  8.129. [Changen](https://github.com/Z-Zheng/Changen) -> code for 2023 paper: Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process

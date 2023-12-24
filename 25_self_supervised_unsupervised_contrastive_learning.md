# 25. Self-supervised, unsupervised & contrastive learning
Self-supervised, unsupervised & contrastive learning are all methods of machine learning that use unlabeled data to train algorithms. Self-supervised learning uses labeled data to create an artificial supervisor, while unsupervised learning uses only the data itself to identify patterns and similarities. Contrastive learning uses pairs of data points to learn representations of data, usually for classification tasks.

  25.1. [Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data](https://devblog.pytorchlightning.ai/seasonal-contrast-transferable-visual-representations-for-remote-sensing-73a17863ed07) -> Seasonal Contrast (SeCo) is an effective pipeline to leverage unlabeled data for in-domain pre-training of remote sensing representations. Models trained with SeCo achieve better performance than their ImageNet pre-trained counterparts and state-of-the-art self-supervised learning methods on multiple downstream tasks. [paper](https://arxiv.org/abs/2103.16607) and [repo](https://github.com/ElementAI/seasonal-contrast)

  25.2. [Unsupervised Learning for Land Cover Classification in Satellite Imagery](https://omdena.com/blog/land-cover-classification/)

  25.3. [Tile2Vec: Unsupervised representation learning for spatially distributed data](https://ermongroup.github.io/blog/tile2vec/)

  25.4. [Contrastive Sensor Fusion](https://github.com/descarteslabs/contrastive_sensor_fusion) -> Code implementing Contrastive Sensor Fusion, an approach for unsupervised learning of multi-sensor representations targeted at remote sensing imagery

  25.5. [hyperspectral-autoencoders](https://github.com/lloydwindrim/hyperspectral-autoencoders) -> Tools for training and using unsupervised autoencoders and supervised deep learning classifiers for hyperspectral data, built on tensorflow. Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction.

  25.6. [Sentinel-2 image clustering in python](https://towardsdatascience.com/sentinel-2-image-clustering-in-python-58f7f2c8a7f6)

  25.7. [MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification](https://arxiv.org/abs/1612.08879) and [code](https://github.com/BUPTLdy/MARTA-GAN)

  25.8. [A generalizable and accessible approach to machine learning with global satellite imagery](https://www.nature.com/articles/s41467-021-24638-z) nature publication -> MOSAIKS is designed to solve an unlimited number of tasks at planet-scale quickly using feature vectors, [with repo](https://github.com/Global-Policy-Lab/mosaiks-paper). Also see [mosaiks-api](https://github.com/calebrob6/mosaiks-api)

  25.9. [contrastive-satellite](https://github.com/hakeemtfrank/contrastive-satellite) -> Using contrastive learning to create embeddings from optical EuroSAT Satellite-2 imagery

  25.10. [Self-Supervised Learning of Remote Sensing Scene Representations Using Contrastive Multiview Coding](https://arxiv.org/abs/2104.07070) -> arxiv paper and [code](https://github.com/vladan-stojnic/CMC-RSSR)

  25.11. [Self-Supervised-Learner by spaceml-org](https://github.com/spaceml-org/Self-Supervised-Learner) -> train a classifier with fewer labeled examples needed using self-supervised learning, example applied to UC Merced land use dataset

  25.12. [deepsentinel](https://github.com/Lkruitwagen/deepsentinel) -> a sentinel-1 and -2 self-supervised sensor fusion model for general purpose semantic embedding

  25.13. [contrastive_SSL_ship_detection](https://github.com/alina2204/contrastive_SSL_ship_detection) -> Contrastive self supervised learning for ship detection in Sentinel 2 images

  25.14. [geography-aware-ssl](https://github.com/sustainlab-group/geography-aware-ssl) -> uses spatially aligned images over time to construct temporal positive pairs in contrastive learning and geo-location to design pre-text tasks

  25.15. [CNN-Supervised Classification](https://github.com/geojames/CNN-Supervised-Classification) -> Python code for self-supervised classification of remotely sensed imagery - part of the Deep Riverscapes project

  25.16. [clustimage](https://github.com/erdogant/clustimage) -> a python package for unsupervised clustering of images

  25.17. [LandSurfaceClustering](https://github.com/lhalloran/LandSurfaceClustering) -> Land surface classification using remote sensing data with unsupervised machine learning (k-means)

  25.18. [K-Means Clustering for Surface Segmentation of Satellite Images](https://medium.com/@maxfieldeland/k-means-clustering-for-surface-segmentation-of-satellite-images-ad1902791ebf)

  25.19. [Sentinel-2 satellite imagery for crop classification using unsupervised clustering](https://medium.com/devseed/sentinel-2-satellite-imagery-for-crop-classification-part-2-47db3745eb49) -> label groups of pixels based on temporal trends of their NDVI values

  25.20. [TheColorOutOfSpace](https://github.com/stevinc/TheColorOutOfSpace) -> Pytorch code for the paper "The color out of space: learning self-supervised representations for Earth Observation imagery" using the BigEarthNet dataset

  25.21. [Semantic segmentation of SAR images using a self supervised technique](https://github.com/cattale93/pytorch_self_supervised_learning)

  25.22. [STEGO](https://github.com/mhamilton723/STEGO) -> Unsupervised Semantic Segmentation by Distilling Feature Correspondences, with [paper](https://arxiv.org/abs/2203.08414)

  25.23. [Unsupervised Segmentation of Hyperspectral Remote Sensing Images with Superpixels](https://github.com/mpBarbato/Unsupervised-Segmentation-of-Hyperspectral-Remote-Sensing-Images-with-Superpixels)

  25.24. [SoundingEarth](https://github.com/khdlr/SoundingEarth) -> Self-supervised Audiovisual Representation Learning for Remote Sensing Data, uses the SoundingEarth [Dataset](https://zenodo.org/record/5600379#.Yom4W5PMK3I)

  25.25. [singleSceneSemSegTgrs2022](https://github.com/sudipansaha/singleSceneSemSegTgrs2022) -> code for 2022 paper: Unsupervised Single-Scene Semantic Segmentation for Earth Observation

  25.26. [SSLRemoteSensing](https://github.com/flyakon/SSLRemoteSensing) -> code for 2021 paper: Semantic Segmentation of Remote Sensing Images With Self-Supervised Multitask Representation Learning

  25.27. [CBT](https://github.com/VMarsocci/CBT) code for 2022 [paper](https://arxiv.org/abs/2205.11319): Continual Barlow Twins: continual self-supervised learning for remote sensing semantic segmentation

  25.28. [Unsupervised Satellite Image Classification based on Partial Adversarial Domain Adaptation](https://github.com/lwpyh/Unsupervised-Satellite-Image-Classfication-based-on-Partial-Domain-Adaptation) -> Code for course project

  25.29. [T2FTS](https://github.com/wdzhao123/T2FTS) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9781379): Teaching Teachers First and Then Student: Hierarchical Distillation to Improve Long-Tailed Object Recognition in Aerial Images

  25.30. [SSLTransformerRS](https://github.com/HSG-AIML/SSLTransformerRS) -> code for 2022 paper: Self-supervised Vision Transformers for Land-cover Segmentation and
  Classification

  25.31. [DINO-MM](https://github.com/zhu-xlab/DINO-MM) -> code for 2022 [paper](https://arxiv.org/abs/2204.05381): Self-supervised Vision Transformers for Joint SAR-optical Representation Learning

  25.32. [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) -> a large-scale dataset for self-supervised learning in Earth observation

  25.33. [SSL4EO-Review](https://github.com/zhu-xlab/SSL4EO-Review) -> code for 2022 [paper](https://arxiv.org/abs/2206.13188): Self-supervised Learning in Remote Sensing: A Review

  25.34. [transfer_learning_cspt](https://github.com/ZhAnGToNG1/transfer_learning_cspt) -> code for 2022 [paper](https://arxiv.org/abs/2207.03860): Consecutive Pretraining: A Knowledge Transfer Learning Strategy with Relevant Unlabeled Data for Remote Sensing Domain

  25.35. [OTL](https://github.com/qlilx/OTL) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/14/3361): Clustering-Based Representation Learning through Output Translation and Its Application to Remote-Sensing Images

  25.36. [Push-and-Pull-Network](https://github.com/WindVChen/Push-and-Pull-Network) -> code for 2022 paper: Contrastive Learning for Fine-grained Ship Classification in Remote Sensing Images

  25.37. [vissl_experiments](https://github.com/lewfish/ssl/tree/main/vissl_experiments) -> Self-supervised Learning using Facebook [VISSL](https://github.com/facebookresearch/vissl) on the RESISC-45 satellite imagery classification dataset

  25.38. [MS2A-Net](https://github.com/Kasra2020/MS2A-Net) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9855229): MS 2 A-Net: Multi-scale spectral-spatial association network for hyperspectral image clustering

  25.39. [UDA_for_RS](https://github.com/Levantespot/UDA_for_RS) -> code for paper: Unsupervised Domain Adaptation for Remote Sensing Semantic Segmentation with Transformer

  25.40. [pytorch-ssl-building_extract](https://github.com/Chendeyue/pytorch-ssl-building_extract) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/21/5350): Research on Self-Supervised Building Information Extraction with High-Resolution Remote Sensing Images for Photovoltaic Potential Evaluation

  25.41. [self-rare-wildlife](https://github.com/xcvil/self-rare-wildlife) -> code for 2021 [paper](https://arxiv.org/abs/2108.07582): Self-Supervised Pretraining and Controlled Augmentation Improve Rare Wildlife Recognition in UAV Images

  25.42. [SatMAE](https://github.com/sustainlab-group/SatMAE) -> code for 2022 [paper](https://arxiv.org/abs/2207.08051): SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery

  25.43. [FireCLR-Wildfires](https://github.com/spaceml-org/FireCLR-Wildfires) -> code for 2022 [paper](https://arxiv.org/abs/2211.14654): Unsupervised Wildfire Change Detection based on Contrastive Learning

  25.44. [FALSE](https://github.com/GeoX-Lab/FALSE) -> code for 2022 [paper](https://arxiv.org/abs/2211.07928): False: False Negative Samples Aware Contrastive Learning for Semantic Segmentation of High-Resolution Remote Sensing Image

  25.45. [MATTER](https://github.com/periakiva/MATTER) -> code for 2022 paper: Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks

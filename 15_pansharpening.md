# 15. Pansharpening

<p align="center">
  <img src="images/pansharpen.png" width="500">
  <br>
  <b>Pansharpening example with a resolution difference of factor 4.</b>
</p>

Pansharpening is a data fusion method that merges the high spatial detail from a high-resolution panchromatic image with the rich spectral information from a lower-resolution multispectral image. The result is a single, high-resolution color image that retains both the sharpness of the panchromatic band and the color information of the multispectral bands. This process enhances the spatial resolution while preserving the spectral qualities of the original images. [Image source](https://www.researchgate.net/publication/308121983_Computer_Vision_for_Large_Format_Digital_Aerial_Cameras)

  15.1. Several algorithms described [in the ArcGIS docs](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/fundamentals-of-panchromatic-sharpening.htm), with the simplest being taking the mean of the pan and RGB pixel value.

  15.2. [PGCU](https://github.com/Zeyu-Zhu/PGCU) -> code for 2023 paper: Probability-based Global Cross-modal Upsampling for Pansharpening

  15.3. [rio-pansharpen](https://github.com/mapbox/rio-pansharpen) -> pansharpening Landsat scenes

  15.4. [Simple-Pansharpening-Algorithms](https://github.com/ThomasWangWeiHong/Simple-Pansharpening-Algorithms)

  15.5. [Working-For-Pansharpening](https://github.com/yuanmaoxun/Working-For-Pansharpening) -> long list of pansharpening methods and update of [Awesome-Pansharpening](https://github.com/Lihui-Chen/Awesome-Pansharpening)

  15.6. [PSGAN](https://github.com/liuqingjie/PSGAN) -> A Generative Adversarial Network for Remote Sensing Image Pan-sharpening, [arxiv paper](https://arxiv.org/abs/1805.03371)

  15.7. [Pansharpening-by-Convolutional-Neural-Network](https://github.com/ThomasWangWeiHong/Pansharpening-by-Convolutional-Neural-Network)

  15.8. [PBR_filter](https://github.com/dbuscombe-usgs/PBR_filter) -> {P}ansharpening by {B}ackground {R}emoval algorithm for sharpening RGB images

  15.9. [py_pansharpening](https://github.com/codegaj/py_pansharpening) -> multiple algorithms implemented in python

  15.10. [Deep-Learning-PanSharpening](https://github.com/xyc19970716/Deep-Learning-PanSharpening) -> deep-learning based pan-sharpening code package, we reimplemented include PNN, MSDCNN, PanNet, TFNet, SRPPNN, and our purposed network DIPNet

  15.11. [HyperTransformer](https://github.com/wgcban/HyperTransformer) -> A Textural and Spectral Feature Fusion Transformer for Pansharpening

  15.12. [DIP-HyperKite](https://github.com/wgcban/DIP-HyperKite) -> Hyperspectral Pansharpening Based on Improved Deep Image Prior and Residual Reconstruction

  15.13. [D2TNet](https://github.com/Meiqi-Gong/D2TNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9761261): A ConvLSTM Network with Dual-direction Transfer for Pan-sharpening

  15.14. [PanColorGAN-VHR-Satellite-Images](https://github.com/esertel/PanColorGAN-VHR-Satellite-Images) -> code for 2020 [paper](https://arxiv.org/abs/2006.16644): Rethinking CNN-Based Pansharpening: Guided Colorization of Panchromatic Images via GANs

  15.15. [MTL_PAN_SEG](https://github.com/andrewekhalel/MTL_PAN_SEG) -> code for 2019 paper: Multi-task deep learning for satellite image pansharpening and segmentation

  15.16. [Z-PNN](https://github.com/matciotola/Z-PNN) -> code for 2022 paper: Pansharpening by convolutional neural networks in the full resolution framework

  15.17. [GTP-PNet](https://github.com/HaoZhang1018/GTP-PNet) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S092427162030352X): GTP-PNet: A residual learning network based on gradient transformation prior for pansharpening

  15.18. [UDL](https://github.com/XiaoXiao-Woo/UDL) -> code for 2021 paper: Dynamic Cross Feature Fusion for Remote Sensing Pansharpening

  15.19. [PSData](https://github.com/yisun98/PSData) -> A Large-Scale General Pan-sharpening DataSet, which contains PSData3 (QB, GF-2, WV-3) and PSData4 (QB, GF-1, GF-2, WV-2).

  15.20. [AFPN](https://github.com/yisun98/AFPN) -> Adaptive Detail Injection-Based Feature Pyramid Network For Pan-sharpening

  15.21. [pan-sharpening](https://github.com/yisun98/pan-sharpening) -> multiple methods demonstrated for multispectral and panchromatic images

  15.22. [PSGan-Family](https://github.com/zhysora/PSGan-Family) -> code for 2020 paper: PSGAN: A Generative Adversarial Network for Remote Sensing Image Pan-Sharpening

  15.23. [PanNet-Landsat](https://github.com/oyam/PanNet-Landsat) -> code for 2017 paper: A Deep Network Architecture for Pan-Sharpening

  15.24. [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) -> code for 2022 paper: Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks

  15.25. [LPPN](https://github.com/ChengJin-git/LPPN) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253521001809): Laplacian pyramid networks: A new approach for multispectral pansharpening

  15.26. [S2_SSC_CNN](https://github.com/hvn2/S2_SSC_CNN) -> code for 2020 paper: Zero-shot Sentinel-2 Sharpening Using A Symmetric Skipped Connection Convolutional Neural Network

  15.27. [S2S_UCNN](https://github.com/hvn2/S2S_UCNN) -> code for 2021 paper: Sentinel 2 sharpening using a single unsupervised convolutional neural network with MTF-Based degradation model

  15.28. [SSE-Net](https://github.com/RSMagneto/SSE-Net) -> code for 2022 paper: Spatial and Spectral Extraction Network With Adaptive Feature Fusion for Pansharpening

  15.29. [UCGAN](https://github.com/zhysora/UCGAN) -> code for 2022 paper: Unsupervised Cycle-consistent Generative Adversarial Networks for Pan-sharpening

  15.30. [GCPNet](https://github.com/Keyu-Yan/GCPNet) -> code for 2022 paper: When Pansharpening Meets Graph Convolution Network and Knowledge Distillation

  15.31. [PanFormer](https://github.com/zhysora/PanFormer) -> code for 2022 [paper](https://arxiv.org/abs/2203.02916): PanFormer: a Transformer Based Model for Pan-sharpening

  15.32. [Pansharpening](https://github.com/nithin-gr/Pansharpening) -> code for 2021 [paper](https://www.researchgate.net/publication/356974466_Pansformers_Transformer-Based_Self-Attention_Network_for_Pansharpening): Pansformers: Transformer-Based Self-Attention Network for Pansharpening

  15.33. [Sentinel-2 Band Pan-Sharpening](https://github.com/purijs/Sentinel-2-Superresolution)

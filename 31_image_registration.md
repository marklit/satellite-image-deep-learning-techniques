# 31. Image registration

Image registration is the process of registering one or more images onto another (typically well georeferenced) image. Traditionally this is performed manually by identifying control points (tie-points) in the images, for example using QGIS. This section lists approaches which mostly aim to automate this manual process. There is some overlap with the data fusion section but the distinction I make is that image registration is performed as a prerequisite to downstream processes which will use the registered data as an input.

  31.1. [Wikipedia article on registration](https://en.wikipedia.org/wiki/Image_registration) -> register for change detection or [image stitching](https://mono.software/2018/03/14/Image-stitching/)

  31.2. [Phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) is used to estimate the XY translation between two images with sub-pixel accuracy. Can be used for accurate registration of low resolution imagery onto high resolution imagery, or to register a [sub-image on a full image](https://www.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html) -> Unlike many spatial-domain algorithms, the phase correlation method is resilient to noise, occlusions, and other defects. With [additional pre-processing](https://scikit-image.org/docs/dev/auto_examples/registration/plot_register_rotation.html) image rotation and scale changes can also be calculated.

  31.3. [How to Co-Register Temporal Stacks of Satellite Images](https://medium.com/sentinel-hub/how-to-co-register-temporal-stacks-of-satellite-images-5167713b3e0b)

  31.4. [ImageRegistration](https://github.com/jandremarais/ImageRegistration) -> Interview assignment for multimodal image registration using SIFT

  31.5. [imreg_dft](https://github.com/matejak/imreg_dft) -> Image registration using discrete Fourier transform. Given two images it can calculate the difference between scale, rotation and position of imaged features. Used by the [up42 co-registration service](https://up42.com/marketplace/blocks/processing/up42-coregistration)

  31.6. [arosics](https://danschef.git-pages.gfz-potsdam.de/arosics/doc/about.html) -> Perform automatic subpixel co-registration of two satellite image datasets using phase-correlation, XY translations only.

  31.7. [SubpixelAlignment](https://github.com/vldkhramtsov/SubpixelAlignment) -> Implementation of tiff image alignment through phase correlation for pixel- and subpixel-bias

  31.8. [cnn-registration](https://github.com/yzhq97/cnn-registration) -> A image registration method using convolutional neural network features written in Python2, Tensorflow 1.5

  31.9. [Detecting Ground Control Points via Convolutional Neural Network for Stereo Matching](https://arxiv.org/abs/1605.02289) -> code?

  31.10. [ImageCoregistration](https://github.com/ily-R/ImageCoregistration) -> Image registration with openCV using sift and RANSAC

  31.11. [mapalignment](https://github.com/Lydorn/mapalignment) -> Aligning and Updating Cadaster Maps with Remote Sensing Images

  31.12. [CVPR21-Deep-Lucas-Kanade-Homography](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography) -> deep learning pipeline to accurately align challenging multimodality images. The method is based on traditional Lucas-Kanade algorithm with feature maps extracted by deep neural networks.

  31.13. [eolearn](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/coregistration/coregistration.html) implements phase correlation, feature matching and [ECC](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/)

  31.14. RStoolbox supports [Image to Image Co-Registration based on Mutual Information](https://bleutner.github.io/RStoolbox/rstbx-docu/coregisterImages.html)

  31.15. [Reprojecting the Perseverance landing footage onto satellite imagery](https://matthewearl.github.io/2021/03/06/mars2020-reproject/)

  31.16. Kornia provides [image registration](https://kornia.readthedocs.io/en/latest/applications/image_registration.html)

  31.17. [LoFTR](https://github.com/zju3dv/LoFTR) -> Detector-Free Local Feature Matching with Transformers. Good performance matching satellite image pairs, tryout the web demo on your data

  31.18. [image-to-db-registration](https://gitlab.orfeo-toolbox.org/remote_modules/image-to-db-registration) -> This remote module implements an algorithm for automated vector Database registration onto an Image. Implemented in the orfeo-toolbox

  31.19. [MS_HLMO_registration](https://github.com/MrPingQi/MS_HLMO_registration) -> Multi-scale Histogram of Local Main Orientation for Remote Sensing Image Registration, with [paper](https://arxiv.org/abs/2204.00260)

  31.20. [cnn-matching](https://github.com/lan-cz/cnn-matching) -> code and datadset for paper: Deep learning algorithm for feature matching of cross modality remote sensing images

  31.21. [Imatch-P](https://github.com/geoyee/Imatch-P) -> A demo using SuperGlue and SuperPoint to do the image matching task based PaddlePaddle

  31.22. [NBR-Net](https://github.com/xuyingxiao/NBR-Net) -> A Non-rigid Bi-directional Registration Network for Multi-temporal Remote Sensing Images

  31.23. [MU-Net](https://github.com/woshiybc/Multi-Scale-Unsupervised-Framework-MSUF) -> code for paper: A Multi-Scale Framework with Unsupervised Learning for Remote Sensing Image Registration

  31.24. [unsupervisedDeepHomographyRAL2018](https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018) -> Unsupervised Deep Homography applied to aerial data

  31.25. [registration_cnn_ntg](https://github.com/zhangliukun/registration_cnn_ntg) -> code for paper: A Multispectral Image Registration Method Based on Unsupervised Learning

  31.26. [remote-sensing-images-registration-dataset](https://github.com/liliangzhi110/remote-sensing-images-registration-dataset) -> at 0.23m, 3.75m & 30m resolution

  31.27. [semantic-template-matching](https://github.com/liliangzhi110/semantictemplatematching) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002446): A deep learning semantic template matching framework for remote sensing image registration

  31.28. [GMN-Generative-Matching-Network](https://github.com/ei1994/GMN-Generative-Matching-Network) -> code for 2018 paper: Deep Generative Matching Network for Optical and SAR Image Registration

  31.29. [SOMatch](https://github.com/system123/SOMatch) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/pii/S0924271620302598): A deep learning framework for matching of SAR and optical imagery

  31.30. [Interspectral image registration dataset](https://medium.com/dronehub/datasets-96fc4f9a92e5) -> including satellite and drone imagery

  31.31. [RISG-image-matching](https://github.com/lan-cz/RISG-image-matching) -> A rotation invariant SuperGlue image matching algorithm

  31.32. [DeepAerialMatching_pytorch](https://github.com/jaehyunnn/DeepAerialMatching_pytorch) -> code for 2020 [paper](https://arxiv.org/abs/2002.01325): A Two-Stream Symmetric Network with Bidirectional Ensemble for Aerial Image Matching

  31.33. [DPCN](https://github.com/ZJU-Robotics-Lab/DPCN) -> code for 2020 [paper](https://arxiv.org/abs/2008.09474): Deep Phase Correlation for End-to-End Heterogeneous Sensor Measurements Matching

  31.34. [FSRA](https://github.com/Dmmm1997/FSRA) -> code for 2022 [paper](https://arxiv.org/abs/2201.09206): A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization

  31.35. [IHN](https://github.com/imdumpl78/IHN) -> code for 2022 [paper](https://arxiv.org/abs/2203.15982): Iterative Deep Homography Estimation

  31.36. [OSMNet](https://github.com/zhanghan9718/OSMNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9609993): Explore Better Network Framework for High-Resolution Optical and SAR Image Matching

  31.37. [L2_Siamese](https://github.com/TheKiteFlier/L2_Siamese) -> code for the 2020 [paper](https://ieeexplore.ieee.org/document/9264687): Registration of Multiresolution Remote Sensing Images Based on L2-Siamese Model

  31.38. [Multi-Step-Deformable-Registration](https://github.com/mpapadomanolaki/Multi-Step-Deformable-Registration) -> code for 2021 paper: Unsupervised Multi-Step Deformable Registration of Remote Sensing Imagery based on Deep Learning

  31.39. [Siamese_ShiftNet](https://github.com/simon-donike/Siamese_ShiftNet) -> NN predicting spatial coregistration shift of remote sensing imagery. Adapted from HighRes-net

# 4. Object detection

<p align="center">
  <img src="images/object-detection.png" width="600">
  <br>
  <b>Image showing the suitability of rotated bounding boxes in remote sensing.</b>
</p>

Object detection in remote sensing involves locating and surrounding objects of interest with bounding boxes. Due to the large size of remote sensing images and the fact that objects may only comprise a few pixels, object detection can be challenging in this context. The imbalance between the area of the objects to be detected and the background, combined with the potential for objects to be easily confused with random features in the background, further complicates the task. Object detection generally performs better on larger objects, but becomes increasingly difficult as the objects become smaller and more densely packed. The accuracy of object detection models can also degrade rapidly as image resolution decreases, which is why it is common to use high resolution imagery, such as 30cm RGB, for object detection in remote sensing. A unique characteristic of aerial images is that objects can be oriented in any direction. To effectively extract measurements of the length and width of an object, it can be crucial to use rotated bounding boxes that align with the orientation of the object. This approach enables more accurate and meaningful analysis of the objects within the image. [Image source](https://www.mdpi.com/2072-4292/13/21/4291)

### 4.1. Object tracking in videos

  4.1.1. [Object Tracking in Satellite Videos Based on a Multi-Frame Optical Flow Tracker](https://arxiv.org/abs/1804.09323) arxiv paper

  4.1.2. [CFME](https://github.com/SY-Xuan/CFME) -> Object Tracking in Satellite Videos by Improved Correlation Filters With Motion Estimations

  4.1.3. [TGraM](https://github.com/HeQibin/TGraM) -> code and dataset for 2022 [paper](https://ieeexplore.ieee.org/document/9715124): Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling

  4.1.4. [satellite_video_mod_groundtruth](https://github.com/zhangjunpeng9354/satellite_video_mod_groundtruth) -> groundtruth on satellite video for evaluating moving object detection algorithm

  4.1.5. [Moving-object-detection-DSFNet](https://github.com/ChaoXiao12/Moving-object-detection-DSFNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9594855): DSFNet: Dynamic and Static Fusion Network for Moving Object Detection in Satellite Videos

  4.1.6. [HiFT](https://github.com/vision4robotics/HiFT) -> code for 2021 [paper](https://arxiv.org/abs/2108.00202): HiFT: Hierarchical Feature Transformer for Aerial Tracking

  4.1.7. [TCTrack](https://github.com/vision4robotics/TCTrack) -> code for 2022 [paper](https://arxiv.org/abs/2203.01885): TCTrack: Temporal Contexts for Aerial Tracking

### 4.2. Object detection with rotated bounding boxes

Orinted bounding boxes (OBB) are polygons representing rotated rectangles. For datasets checkout DOTA & HRSC2016

  4.2.1. [mmrotate](https://github.com/open-mmlab/mmrotate) -> Rotated Object Detection Benchmark, with pretrained models and function for inferencing on very large images

  4.2.2. [OBBDetection](https://github.com/jbwang1997/OBBDetection) -> an oriented object detection library, which is based on MMdetection

  4.2.3. [rotate-yolov3](https://github.com/ming71/rotate-yolov3) -> Rotation object detection implemented with yolov3. Also see [yolov3-polygon](https://github.com/ming71/yolov3-polygon)

  4.2.4. [DRBox](https://github.com/liulei01/DRBox) -> for detection tasks where the objects are orientated arbitrarily, e.g. vehicles, ships and airplanes

  4.2.5. [s2anet](https://github.com/csuhan/s2anet) -> Official code of the paper 'Align Deep Features for Oriented Object Detection'

  4.2.6. [CFC-Net](https://github.com/ming71/CFC-Net) -> Official implementation of "CFC-Net: A Critical Feature Capturing Network for Arbitrary-Oriented Object Detection in Remote Sensing Images"

  4.2.7. [ReDet](https://github.com/csuhan/ReDet) -> Official code of the paper "ReDet: A Rotation-equivariant Detector for Aerial Object Detection"

  4.2.8. [BBAVectors-Oriented-Object-Detection](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection) -> Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors

  4.2.9. [CSL_RetinaNet_Tensorflow](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow) -> Code for ECCV 2020 paper: Arbitrary-Oriented Object Detection with Circular Smooth Label

  4.2.10. [r3det-on-mmdetection](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection) -> R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object

  4.2.11. [R-DFPN_FPN_Tensorflow](https://github.com/yangxue0827/R-DFPN_FPN_Tensorflow) -> Rotation Dense Feature Pyramid Networks (Tensorflow)

  4.2.12. [R2CNN_Faster-RCNN_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow) -> Rotational region detection based on Faster-RCNN

  4.2.13. [Rotated-RetinaNet](https://github.com/ming71/Rotated-RetinaNet) -> implemented in pytorch, it supports the following datasets: DOTA, HRSC2016, ICDAR2013, ICDAR2015, UCAS-AOD, NWPU VHR-10, VOC2007

  4.2.14. [OBBDet_Swin](https://github.com/ming71/OBBDet_Swin) -> The sixth place winning solution in 2021 Gaofen Challenge

  4.2.15. [CG-Net](https://github.com/WeiZongqi/CG-Net) -> Learning Calibrated-Guidance for Object Detection in Aerial Images. With [paper](https://ieeexplore.ieee.org/abstract/document/9735375)

  4.2.16. [OrientedRepPoints_DOTA](https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA) -> Oriented RepPoints + Swin Transformer/ReResNet

  4.2.17. [yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb) -> yolov5 + Oriented Object Detection

  4.2.18. [How to Train YOLOv5 OBB](https://blog.roboflow.com/yolov5-for-oriented-object-detection/) -> YOLOv5 OBB tutorial and [YOLOv5 OBB noteboook](https://colab.research.google.com/drive/16nRwsioEYqWFLBF5VpT_NvELeOeupURM#scrollTo=1NZxhXTMWvek)

  4.2.19. [OHDet_Tensorflow](https://github.com/SJTU-Thinklab-Det/OHDet_Tensorflow) -> can be applied to rotation detection and object heading detection

  4.2.20. [Seodore](https://github.com/nijkah/Seodore) -> framework maintaining recent updates of mmdetection

  4.2.21. [Rotation-RetinaNet-PyTorch](https://github.com/HsLOL/Rotation-RetinaNet-PyTorch) -> oriented detector Rotation-RetinaNet implementation on Optical and SAR ship dataset

  4.2.22. [AIDet](https://github.com/jwwangchn/aidet) -> an open source object detection in aerial image toolbox based on MMDetection

  4.2.23. [rotation-yolov5](https://github.com/BossZard/rotation-yolov5) -> rotation detection based on yolov5

  4.2.24. [ShipDetection](https://github.com/lilinhao/ShipDetection) -> Ship Detection in HR Optical Remote Sensing Images via Rotated Bounding Box, based on Faster R-CNN and ORN, uses caffe

  4.2.25. [SLRDet](https://github.com/LUCKMOONLIGHT/SLRDet) -> project based on mmdetection to reimplement RRPN and use the model Faster R-CNN OBB

  4.2.26. [AxisLearning](https://github.com/RSIA-LIESMARS-WHU/AxisLearning) -> code for 2020 [paper](https://www.mdpi.com/2072-4292/12/6/908): Axis Learning for Orientated Objects Detection in Aerial Images

  4.2.27. [Detection_and_Recognition_in_Remote_Sensing_Image](https://github.com/whywhs/Detection_and_Recognition_in_Remote_Sensing_Image) -> This work uses PaNet to realize Detection and Recognition in Remote Sensing Image by MXNet

  4.2.28. [DrBox-v2-tensorflow](https://github.com/ZongxuPan/DrBox-v2-tensorflow) -> tensorflow implementation of DrBox-v2 which is an improved detector with rotatable boxes for target detection in remote sensing images

  4.2.29. [Rotation-EfficientDet-D0](https://github.com/HsLOL/Rotation-EfficientDet-D0) -> A PyTorch Implementation Rotation Detector based EfficientDet Detector, applied to custom rotation vehicle datasets

  4.2.30. [DODet](https://github.com/yanqingyao1994/DODet) -> Dual alignment for oriented object detection, uses DOTA dataset. With [paper](https://ieeexplore.ieee.org/abstract/document/9706434)

  4.2.31. [GF-CSL](https://github.com/WangJian981002/GF-CSL) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9776580): Gaussian Focal Loss: Learning Distribution Polarized Angle Prediction for Rotated Object Detection in Aerial Images

  4.2.32. [simplified_rbox_cnn](https://github.com/SIAnalytics/simplified_rbox_cnn) -> code for 2018 [paper](https://dl.acm.org/doi/10.1145/3274895.3274915): RBox-CNN: rotated bounding box based CNN for ship detection in remote sensing image. Uses Tensorflow object detection API

  4.2.33. [Polar-Encodings](https://github.com/flyingshan/Learning-Polar-Encodings-For-Arbitrary-Oriented-Ship-Detection-In-SAR-Images) -> code for 2021 [paper](Learning Polar Encodings for Arbitrary-Oriented Ship Detection in SAR Images)

  4.2.34. [R-CenterNet](https://github.com/ZeroE04/R-CenterNet) -> detector for rotated-object based on CenterNet

  4.2.35. [piou](https://github.com/clobotics/piou) -> Orientated Object Detection; IoU Loss, applied to DOTA dataset

  4.2.36. [DAFNe](https://github.com/steven-lang/DAFNe) -> code for 2021 [paper](https://arxiv.org/abs/2109.06148): DAFNe: A One-Stage Anchor-Free Approach for Oriented Object Detection

  4.2.37. [AProNet](https://github.com/geovsion/AProNet) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S092427162100229X): AProNet: Detecting objects with precise orientation from aerial images. Applied to datasets DOTA and HRSC2016

  4.2.38. [UCAS-AOD-benchmark](https://github.com/ming71/UCAS-AOD-benchmark) -> A benchmark of UCAS-AOD dataset

  4.2.39. [RotateObjectDetection](https://github.com/XinzeLee/RotateObjectDetection) -> based on Ultralytics/yolov5, with adjustments to enable rotate prediction boxes. Also see [PolygonObjectDetection](https://github.com/XinzeLee/PolygonObjectDetection)

  4.2.40. [AD-Toolbox](https://github.com/liuyanyi/AD-Toolbox) -> Aerial Detection Toolbox based on MMDetection and MMRotate, with support for more datasets

  4.2.41. [GGHL](https://github.com/Shank2358/GGHL) -> code for 2022 [paper](https://arxiv.org/abs/2109.12848): A General Gaussian Heatmap Label Assignment for Arbitrary-Oriented Object Detection

  4.2.42. [NPMMR-Det](https://github.com/Shank2358/NPMMR-Det) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9364888): A Novel Nonlocal-Aware Pyramid and Multiscale Multitask Refinement Detector for Object Detection in Remote Sensing Images

  4.2.43. [AOPG](https://github.com/jbwang1997/AOPG) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9795321): Anchor-Free Oriented Proposal Generator for Object Detection

  4.2.44. [SE2-Det](https://github.com/Virusxxxxxxx/SE2-Det) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/15/3637): Semantic-Edge-Supervised Single-Stage Detector for Oriented Object Detection in Remote Sensing Imagery

  4.2.45. [OrientedRepPoints](https://github.com/LiWentomng/OrientedRepPoints) -> code for 2021 [paper](https://arxiv.org/abs/2105.11111): Oriented RepPoints for Aerial Object Detection

  4.2.46. [TS-Conv](https://github.com/Shank2358/TS-Conv) -> code for 2022 [paper](https://arxiv.org/abs/2209.02200): Task-wise Sampling Convolutions for Arbitrary-Oriented Object Detection in Aerial Images

  4.2.47. [FCOSR](https://github.com/lzh420202/FCOSR) -> A Simple Anchor-free Rotated Detector for Aerial Object Detection. This implement is modified from mmdetection. See also [TensorRT_Inference](https://github.com/lzh420202/TensorRT_Inference)

  4.2.48. [OBB_Detection](https://github.com/HsLOL/OBB_Detection) -> Finalist's solution in the track of Oriented Object Detection in Remote Sensing Images, 2022 Guangdong-Hong Kong-Macao Greater Bay Area International Algorithm Competition

  4.2.49. [sam-mmrotate](https://github.com/Li-Qingyun/sam-mmrotate) -> SAM (Segment Anything Model) for generating rotated bounding boxes with MMRotate, which is a comparison method of H2RBox-v2

  4.2.50. [mmrotate-dcfl](https://github.com/Chasel-Tsui/mmrotate-dcfl) -> code for 2023 paper: Dynamic Coarse-to-Fine Learning for Oriented Tiny Object Detection

  4.2.51. [h2rbox-mmrotate](https://github.com/yangxue0827/h2rbox-mmrotate) -> code for 2022 paper: H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection

  4.2.52. [Spatial-Transform-Decoupling](https://github.com/yuhongtian17/Spatial-Transform-Decoupling) -> code for 2023 paper: Spatial Transform Decoupling for Oriented Object Detection

  4.2.53. [ARS-DETR](https://github.com/httle/ARS-DETR) -> code for 2023 paper: ARS-DETR: Aspect Ratio Sensitive Oriented Object Detection with Transformer

  4.2.54. [CFINet](https://github.com/shaunyuan22/CFINet) -> code for 2023 paper: Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning. Introduces [SODA-A dataset](https://shaunyuan22.github.io/SODA/)

### 4.3. Object detection enhanced by super resolution

  4.3.1. [Super-Resolution and Object Detection](https://medium.com/the-downlinq/super-resolution-and-object-detection-a-love-story-part-4-8ad971eef81e) -> Super-resolution is a relatively inexpensive enhancement that can improve object detection performance

  4.3.2. [EESRGAN](https://github.com/Jakaria08/EESRGAN) -> Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network

  4.3.3. [Mid-Low Resolution Remote Sensing Ship Detection Using Super-Resolved Feature Representation](https://www.preprints.org/manuscript/202108.0337/v1)

  4.3.4. [EESRGAN](https://github.com/divyam96/EESRGAN) -> code for 2020 [paper](https://www.mdpi.com/2072-4292/12/9/1432): Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network. Applied to COWC & [OGST](https://data.mendeley.com/datasets/bkxj8z84m9/3) datasets

  4.3.5. [FBNet](https://github.com/wdzhao123/FBNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9739789): Feature Balance for Fine-Grained Object Classification in Aerial Images

  4.3.6. [SuperYOLO](https://github.com/icey-zhang/SuperYOLO) -> code for 2022 [paper](https://arxiv.org/abs/2209.13351): SuperYOLO: Super Resolution Assisted Object Detection in Multimodal Remote Sensing Imagery

### 4.4. Salient object detection

Detecting the most noticeable or important object in a scene

  4.4.1. [ACCoNet](https://github.com/MathLee/ACCoNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9756652): Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images

  4.4.2. [MCCNet](https://github.com/MathLee/MCCNet) -> Multi-Content Complementation Network for Salient Object Detection in Optical Remote Sensing Images

  4.4.3. [CorrNet](https://github.com/MathLee/CorrNet) -> Lightweight Salient Object Detection in Optical Remote Sensing Images via Feature Correlation. With [paper](https://arxiv.org/abs/2201.08049)

  4.4.4. [Reading list for deep learning based Salient Object Detection in Optical Remote Sensing Images](https://github.com/MathLee/ORSI-SOD_Summary)

  4.4.5. [ORSSD-dataset](https://github.com/rmcong/ORSSD-dataset) -> salient object detection dataset

  4.4.6. [EORSSD-dataset](https://github.com/rmcong/EORSSD-dataset) -> Extended Optical Remote Sensing Saliency Detection (EORSSD) Dataset

  4.4.7. [DAFNet_TIP20](https://github.com/rmcong/DAFNet_TIP20) -> code for 2020 [paper](https://arxiv.org/abs/2011.13144): Dense Attention Fluid Network for Salient Object Detection in Optical Remote Sensing Images

  4.4.8. [EMFINet](https://github.com/Kunye-Shen/EMFINet) -> code for 2021 paper: Edge-Aware Multiscale Feature Integration Network for Salient Object Detection in Optical Remote Sensing Images

  4.4.9. [ERPNet](https://github.com/zxforchid/ERPNet) -> code for 2022 paper: Edge-guided Recurrent Positioning Network for Salient Object Detection in Optical Remote Sensing Images

  4.4.10. [FSMINet](https://github.com/zxforchid/FSMINet) -> code for 2022 paper: Fully Squeezed Multi-Scale Inference Network for Fast and Accurate Saliency Detection in Optical Remote Sensing Images

  4.4.11. [AGNet](https://github.com/NuaaYH/AGNet) -> code for 2022 paper: AGNet: Attention Guided Network for Salient Object Detection in Optical Remote Sensing Images

  4.4.12. [MSCNet](https://github.com/NuaaYH/MSCNet) -> code for 2022 [paper](https://arxiv.org/abs/2205.08959): A lightweight multi-scale context network for salient object detection in optical remote sensing images

  4.4.13. [GPnet](https://github.com/liuyu1002/GPnet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9687549): Global Perception Network for Salient Object Detection in Remote Sensing Images

  4.4.14. [SeaNet](https://github.com/MathLee/SeaNet) -> code for 2023 [paper](https://arxiv.org/abs/2301.02778): Lightweight Salient Object Detection in Optical Remote Sensing Images via Semantic Matching and Edge Alignment

  4.4.15. [GeleNet](https://github.com/MathLee/GeleNet) -> code for 2023 paper: Salient Object Detection in Optical Remote Sensing Images Driven by Transformer

### 4.5. Object detection - Buildings, rooftops & solar panels

  4.5.1. [satellite_image_tinhouse_detector](https://github.com/yasserius/satellite_image_tinhouse_detector) -> Detection of tin houses from satellite/aerial images using the Tensorflow Object Detection API `BEGINNER`

  4.5.2. [Machine Learning For Rooftop Detection and Solar Panel Installment](https://omdena.com/blog/machine-learning-rooftops/) discusses tiling large images and generating annotations from OSM data. Features of the roofs were calculated using a combination of contour detection and classification. [Follow up article using semantic segmentation](https://omdena.com/blog/rooftops-classification/)

  4.5.3. [Building Extraction with YOLT2 and SpaceNet Data](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)

  4.5.4. [XBD-hurricanes](https://github.com/dbuscombe-usgs/XBD-hurricanes) -> Models for building (and building damage) detection in high-resolution (<1m) satellite and aerial imagery using a modified RetinaNet model

  4.5.5. [Detecting solar panels from satellite imagery](https://towardsdatascience.com/weekend-project-detecting-solar-panels-from-satellite-imagery-f6f5d5e0da40) using segmentation

  4.5.6. [ssd-spacenet](https://github.com/aurotripathy/ssd-spacenet) -> Detect buildings in the Spacenet dataset using Single Shot MultiBox Detector (SSD)

  4.5.7. [3DBuildingInfoMap](https://github.com/LllC-mmd/3DBuildingInfoMap) -> simultaneous extraction of building height and footprint from Sentinel imagery using ResNet

  4.5.8. [Solar Panel Detection](https://medium.com/analytics-vidhya/solar-panel-detection-from-aerial-view-or-satellite-images-648c22c260ba) -> using Faster R-CNN & Tensorflow object detection API. With [repo](https://github.com/shiva2410/Solar_Panel-Detection-in-Aerial-Images)

  4.5.9. [DeepSolaris](https://github.com/thinkpractice/DeepSolaris) -> a EuroStat project to detect solar panels in aerial images, further material [here](https://github.com/FHNW-IVGI/workshop_geopython2019/tree/master/Ex.02_SolarPanels)

  4.5.10. [ML_ObjectDetection_CAFO](https://github.com/Qberto/ML_ObjectDetection_CAFO) -> Detect Concentrated Animal Feeding Operations (CAFO) in Satellite Imagery

  4.5.11. [Multi-level-Building-Detection-Framework](https://github.com/luoxiaoliaolan/Multi-level-Building-Detection-Framework) -> code for 2018 [paper](https://ieeexplore.ieee.org/document/8458225): Multilevel Building Detection Framework in Remote Sensing Images Based on Convolutional Neural Networks

  4.5.12. [Automatic Damage Annotation on Post-Hurricane Satellite Imagery](https://dds-lab.github.io/disaster-damage-detection/) -> detect damaged buildings using tensorflow object detection API. With repos [here](https://github.com/DDS-Lab/disaster-image-processing) and [here](https://github.com/annieyan/PreprocessSatelliteImagery-ObjectDetection)

  4.5.13. [mappingchallenge](https://github.com/krishanr/mappingchallenge) -> YOLOv5 applied to the AICrowd Mapping Challenge dataset

### 4.6. Object detection - Ships & boats

  4.6.1. [kaggle-ships-in-Google-Earth-yolov5](https://github.com/robmarkcole/kaggle-ships-in-Google-Earth-yolov5) -> Applying YOLOv5 to Kaggle Ships in Google Earth dataset `BEGINNER`

  4.6.2. [How hard is it for an AI to detect ships on satellite images?](https://medium.com/earthcube-stories/how-hard-it-is-for-an-ai-to-detect-ships-on-satellite-images-7265e34aadf0)

  4.6.3. [Object Detection in Satellite Imagery, a Low Overhead Approach](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7)

  4.6.4. [Detecting Ships in Satellite Imagery](https://medium.com/dataseries/detecting-ships-in-satellite-imagery-7f0ca04e7964) using the Planet dataset and Keras

  4.6.5. [Planet use non DL felzenszwalb algorithm to detect ships](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ship-detector/01_ship_detector.ipynb)

  4.6.6. [Ship detection using k-means clustering & CNN classifier on patches](https://towardsdatascience.com/data-science-and-satellite-imagery-985229e1cd2f)

  4.6.7. [SARfish](https://github.com/MJCruickshank/SARfish) -> Ship detection in Sentinel 1 Synthetic Aperture Radar (SAR) imagery

  4.6.8. [Arbitrary-Oriented Ship Detection through Center-Head Point Extraction](https://arxiv.org/abs/2101.11189) -> arxiv paper. Keypoint estimation is performed to find the center of ships. Then, the size and head point of the ships are regressed. Repo [ASD](https://github.com/JinleiMa/ASD)

  4.6.9. [ship_detection](https://github.com/rugg2/ship_detection) -> using an interesting combination of CNN classifier, Class Activation Mapping (CAM) & UNET segmentation. Accompanying [three part blog post](https://www.vortexa.com/insights/technology/satellite-images-object-detection/)

  4.6.10. [Building a complete Ship detection algorithm using YOLOv3 and Planet satellite images](https://medium.com/intel-software-innovators/ship-detection-in-satellite-images-from-scratch-849ccfcc3072) -> covers finding and annotating data (using LabelMe), preprocessing large images into chips, and training Yolov3. [Repo](https://github.com/amanbasu/ship-detection)

  4.6.11. [Ship-detection-in-satellite-images](https://github.com/zmf0507/Ship-detection-in-satellite-images) -> experiments with  UNET, YOLO, Mask R-CNN, SSD, Faster R-CNN, RETINA-NET

  4.6.12. [Ship-Detection-from-Satellite-Images-using-YOLOV4](https://github.com/debasis-dotcom/Ship-Detection-from-Satellite-Images-using-YOLOV4) -> uses Kaggle Airbus Ship Detection dataset

  4.6.13. [kaggle-airbus-ship-detection-challenge](https://github.com/toshi-k/kaggle-airbus-ship-detection-challenge) -> using oriented SSD

  4.6.14. [shipsnet-detector](https://github.com/rhammell/shipsnet-detector) -> Detect container ships in Planet imagery using machine learning

  4.6.15. [Classifying Ships in Satellite Imagery with Neural Networks](https://towardsdatascience.com/classifying-ships-in-satellite-imagery-with-neural-networks-944024879651) -> applied to the Kaggle Ships in Satellite Imagery dataset

  4.6.16. [Mask R-CNN for Ship Detection & Segmentation](https://medium.com/@gabogarza/mask-r-cnn-for-ship-detection-segmentation-a1108b5a083) blog post with [repo](https://github.com/gabrielgarza/Mask_RCNN)

  4.6.17. [contrastive_SSL_ship_detection](https://github.com/alina2204/contrastive_SSL_ship_detection) -> Contrastive self supervised learning for ship detection in Sentinel 2 images

  4.6.18. [Boat detection with multi-region-growing method in satellite images](https://medium.com/@ipmach/boat-detection-with-multi-region-growing-method-in-satellite-images-3339a6c29a8c)

  4.6.19. [small-boat-detector](https://github.com/swricci/small-boat-detector) -> Trained yolo v3 model weights and configuration file to detect small boats in satellite imagery

  4.6.20. [Satellite-Imagery-Datasets-Containing-Ships](https://github.com/JasonManesis/Satellite-Imagery-Datasets-Containing-Ships) -> A list of optical and radar satellite datasets for ship detection, classification, semantic segmentation and instance segmentation tasks

  4.6.21. [vessel-detection-sentinels](https://github.com/allenai/vessel-detection-sentinels) -> Sentinel-1 and Sentinel-2 Vessel Detection

  4.6.22. [Ship-Detection](https://github.com/gouravbarkle/Ship-Detection) -> CNN approach for ship detection in the ocean using a satellite image

  4.6.23. [vesselTracker](https://github.com/carlossantamarizq/vesselTracker) -> Project based on reduced model of Yolov5 architecture using Pytorch. Custom dataset based on SAR imagery provided by Sentinel-1 through Earth Engine API

  4.6.24. [marine-debris-ml-model](https://github.com/danieltyukov/marine-debris-ml-model) -> Marine Debris Detection using tensorflow object detection API

  4.6.25. [SDGH-Net](https://github.com/WangZhenqing-RS/SDGH-Net-Ship-Detection-in-Optical-Remote-Sensing-Images-Based-on-Gaussian-Heatmap-Regression) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/3/499): SDGH-Net: Ship Detection in Optical Remote Sensing Images Based on Gaussian Heatmap Regression

  4.6.26. [LR-TSDet](https://github.com/Lausen-Ng/LR-TSDet) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/19/3890): LR-TSDet: Towards Tiny Ship Detection in Low-Resolution Remote Sensing Images

  4.6.27. [FGSCR-42](https://github.com/DYH666/FGSCR-42) -> A public Dataset for Fine-Grained Ship Classification in Remote sensing images

  4.6.28. [ShipDetection](https://github.com/lilinhao/ShipDetection) -> Ship Detection in HR Optical Remote Sensing Images via Rotated Bounding Box, based on Faster R-CNN and ORN, uses caffe

  4.6.29. [WakeNet](https://github.com/Lilytopia/WakeNet) -> A CNN-based optical image ship wake detector, code for 2021 paper: Rethinking Automatic Ship Wake Detection: State-of-the-Art CNN-based Wake Detection via Optical Images

  4.6.30. [Histogram of Oriented Gradients (HOG) Boat Heading Classification](https://medium.com/the-downlinq/histogram-of-oriented-gradients-hog-heading-classification-a92d1cf5b3cc) -> Medium article

  4.6.31. [Object Detection in Satellite Imagery, a Low Overhead Approach](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7) -> Medium article which demonstrates how to combine Canny edge detector pre-filters with HOG feature descriptors, random forest classifiers, and sliding windows to perform ship detection

  4.6.32. [simplified_rbox_cnn](https://github.com/SIAnalytics/simplified_rbox_cnn) -> code for 2018 [paper](https://dl.acm.org/doi/10.1145/3274895.3274915): RBox-CNN: rotated bounding box based CNN for ship detection in remote sensing image. Uses Tensorflow object detection API

  4.6.33. [Ship-Detection-based-on-YOLOv3-and-KV260](https://github.com/xlsjdjdk/Ship-Detection-based-on-YOLOv3-and-KV260) -> entry project of the Xilinx Adaptive Computing Challenge 2021. It uses YOLOv3 for ship target detection in optical remote sensing images, and deploys DPU on the KV260 platform to achieve hardware acceleration

  4.6.34. [LEVIR-Ship](https://github.com/WindVChen/LEVIR-Ship) -> a dataset for tiny ship detection under medium-resolution remote sensing images

  4.6.35. [Push-and-Pull-Network](https://github.com/WindVChen/Push-and-Pull-Network) -> code for 2022 paper: Contrastive Learning for Fine-grained Ship Classification in Remote Sensing Images

  4.6.36. [DRENet](https://github.com/WindVChen/DRENet) -> code for 2022 [paper])(https://ieeexplore.ieee.org/abstract/document/9791363): A Degraded Reconstruction Enhancement-Based Method for Tiny Ship Detection in Remote Sensing Images With a New Large-Scale Dataset

  4.6.37. [xView3-The-First-Place-Solution](https://github.com/BloodAxe/xView3-The-First-Place-Solution) - A winning solution for [xView 3](https://iuu.xview.us/) challenge (Vessel detection, classification and length estimation on Sentinetl-1 images). Contains trained models, inference pipeline and training code & configs to reproduce the results.

  4.6.38. [vessel-detection-viirs](https://github.com/allenai/vessel-detection-viirs) -> Model and service code for streaming vessel detections from VIIRS satellite imagery

  4.6.39. [anomaly-detection-in-SAR-imagery](https://github.com/iamyadavabhishek/anomaly-detection-in-SAR-imagery) -> identify an unknown ship in docks using keras & retinanet

### 4.7. Object detection - Cars, vehicles & trains

  4.7.1. [Detection of parkinglots and driveways with retinanet](https://github.com/spiyer99/retinanet) `BEGINNER`

  4.7.2. [pytorch-vedai](https://github.com/MichelHalmes/pytorch-vedai) -> object detection on the VEDAI dataset: Vehicle Detection in Aerial Imagery `BEGINNER`

  4.7.3. [Truck Detection with Sentinel-2 during COVID-19 crisis](https://github.com/hfisser/Truck_Detection_Sentinel2_COVID19) -> moving objects in Sentinel-2 data causes a specific reflectance relationship in the RGB, which looks like a rainbow, and serves as a marker for trucks. Improve accuracy by only analysing roads. Not using object detection but relevant. Also see [S2TD](https://github.com/hfisser/S2TD)

  4.7.4. [cowc_car_counting](https://github.com/motokimura/cowc_car_counting) -> car counting on the [Cars Overhead With Context (COWC) dataset](https://gdo152.llnl.gov/cowc/). Not sctictly object detection but a CNN to predict the car count in a tile

  4.7.5. [CarCounting](https://github.com/JacksonPeoples/CarCounting) -> using Yolov3 & COWC dataset

  4.7.6. [Traffic density estimation as a regression problem instead of object detection](https://omdena.com/blog/ai-road-safety/) -> inspired by [this paper](https://ieeexplore.ieee.org/document/8916990)

  4.7.7. [Applying Computer Vision to Railcar Detection](https://orbitalinsight.com/blog/apping-computer-vision-to-railcar-detection) -> useful insights into counting railcars (i.e. train carriages) using Mask-RCNN with rotated bounding boxes output

  4.7.8. [Leveraging Deep Learning for Vehicle Detection And Classification](https://orbitalinsight.com/blog/leveraging-deep-learning-for-vehicle-detection-and-classification)

  4.7.9. [Rotation-EfficientDet-D0](https://github.com/HsLOL/Rotation-EfficientDet-D0) -> PyTorch implementation of Rotated EfficientDet, applied to a custom rotation vehicle dataset (car counting)

  4.7.10. [RSVC2021-Dataset](https://github.com/YinongGuo/RSVC2021-Dataset) -> A dataset for Vehicle Counting in Remote Sensing images, created from the DOTA & ITCVD

  4.7.11. [Car Localization and Counting with Overhead Imagery, an Interactive Exploration](https://medium.com/the-downlinq/car-localization-and-counting-with-overhead-imagery-an-interactive-exploration-9d5a029a596b) -> Medium article by Adam Van Etten

  4.7.12. [Vehicle-Counting-in-Very-Low-Resolution-Aerial-Images](https://github.com/hbsszq/Vehicle-Counting-in-Very-Low-Resolution-Aerial-Images) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9775767): Vehicle Counting in Very Low-Resolution Aerial Images via Cross-Resolution Spatial Consistency and Intraresolution Time Continuity

  4.7.13. [Vehicle Detection blog post](https://www.silvispace.xyz/posts/vehicle-post/) by Grant Pearse: detecting vehicles across New Zealand without collecting local training data

### 4.8. Object detection - Planes & aircraft

  4.8.1. [Faster RCNN to detect airplanes](https://github.com/ShubhankarRawat/Airplane-Detection-for-Satellites) `BEGINNER`

  4.8.2. [yoltv4](https://github.com/avanetten/yoltv4) includes examples on the [RarePlanes dataset](https://registry.opendata.aws/rareplanes/)

  4.8.3. [aircraft-detection](https://github.com/hakeemtfrank/aircraft-detection) -> experiments to test the performance of a Gaussian process (GP) classifier with various kernels on the UC Merced land use land cover (LULC) dataset

  4.8.4. [Using Detectron2 to segment aircraft from satellite imagery](https://share.buitrongan.com/using-detectron2-to-segments-aircraft-from-satellite-images-5a8ac1a0d35e) -> pytorch and Rare Planes

  4.8.5. [aircraft-detection-from-satellite-images-yolov3](https://github.com/emrekrtorun/aircraft-detection-from-satellite-images-yolov3) -> trained on kaggle cgi-planes-in-satellite-imagery-w-bboxes dataset

  4.8.6. [HRPlanesv2-Data-Set](https://github.com/dilsadunsal/HRPlanesv2-Data-Set) -> YOLOv4 and YOLOv5 weights trained on the HRPlanesv2 dataset

  4.8.7. [Deep-Learning-for-Aircraft-Recognition](https://github.com/Shayan-Bravo/Deep-Learning-for-Aircraft-Recognition) -> A CNN model trained to classify and identify various military aircraft through satellite imagery

  4.8.8. [FRCNN-for-Aircraft-Detection](https://github.com/Huatsing-Lau/FRCNN-for-Aircraft-Detection) -> faster-rcnn & keras

  4.8.9. [ergo-planes-detector](https://github.com/evilsocket/ergo-planes-detector) -> An ergo based project that relies on a convolutional neural network to detect airplanes from satellite imagery, uses the PlanesNet dataset

  4.8.10. [pytorch-remote-sensing](https://github.com/miko7879/pytorch-remote-sensing) -> Aircraft detection using the 'Airbus Aircraft Detection' dataset and Faster-RCNN with ResNet-50 backbone using pytorch

  4.8.11. [FasterRCNN_ObjectDetection](https://github.com/UKMIITB/FasterRCNN_ObjectDetection) -> faster RCNN model for aircraft detection and localisation in satellite images and creating a webpage with live server for public usage

  4.8.12. [HRPlanes](https://github.com/TolgaBkm/HRPlanes) -> weights of YOLOv4 and Faster R-CNN networks trained with HRPlanes dataset

  4.8.13. [aerial-detection](https://github.com/alexbakr/aerial-detection) -> uses Yolov5 & Icevision

  4.8.14. [How to choose a deep learning architecture to detect aircrafts in satellite imagery?](https://medium.com/artificialis/how-to-choose-a-deep-learning-model-to-detect-aircrafts-in-satellite-imagery-cd7d106e76ad)

  4.8.15. [rareplanes-yolov5](https://github.com/jeffaudi/rareplanes-yolov5) -> using YOLOv5 and the RarePlanes dataset to detect and classify sub-characteristics of aircraft, with [article](https://medium.com/artificialis/detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5-5f3d464b75ad)

  4.8.16. [OnlyPlanes](https://github.com/naivelogic/OnlyPlanes) -> dataset and pretrained models for the paper: OnlyPlanes - Incrementally Tuning Synthetic Training Datasets for Satellite Object Detection

  4.8.17. [Understanding the RarePlanes Dataset and Building an Aircraft Detection Model](https://encord.com/blog/rareplane-dataset-aircraft-detection-model/) -> blog post

### 4.9. Object detection - Infrastructure & utilities

  4.9.1. [wind-turbine-detector](https://github.com/lbborkowski/wind-turbine-detector) -> Wind Turbine Object Detection from Aerial Imagery Using TensorFlow Object Detection API

  4.9.2. [Water Tanks and Swimming Pools Detection](https://github.com/EduardoFernandes1410/PATREO-Dengue) -> uses Faster R-CNN

  4.9.3. [PCAN](https://www.mdpi.com/2072-4292/13/7/1243) -> Part-Based Context Attention Network for Thermal Power Plant Detection in Remote Sensing Imagery, with [dataset](https://github.com/wenxinYin/AIR-TPPDD)

### 4.10. Object detection - Oil storage tank detection

Oil is stored in tanks at many points between extraction and sale, and the volume of oil in storage is an important economic indicator.

  4.10.1. [A Beginner’s Guide To Calculating Oil Storage Tank Occupancy With Help Of Satellite Imagery](https://medium.com/planet-stories/a-beginners-guide-to-calculating-oil-storage-tank-occupancy-with-help-of-satellite-imagery-e8f387200178)

  4.10.2. [Oil Storage Tank’s Volume Occupancy On Satellite Imagery Using YoloV3](https://towardsdatascience.com/oil-storage-tanks-volume-occupancy-on-satellite-imagery-using-yolov3-3cf251362d9d) with [repo](https://github.com/mdmub0587/Oil-Storage-Tank-s-Volume-Occupancy)

  4.10.3. [Oil-Tank-Volume-Estimation](https://github.com/kheyer/Oil-Tank-Volume-Estimation) -> combines object detection and classical computer vision

  4.10.4. [Oil tank instance segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f) using Keras & Airbus Oil Storage Detection Dataset on Kaggle

  4.10.5. https://www.kaggle.com/towardsentropy/oil-storage-tanks -> large kaggle dataset, note however that approx 85% of images contain no tanks

  4.10.6. https://www.kaggle.com/airbusgeo/airbus-oil-storage-detection-dataset -> smaller kaggle dataset

  4.10.7. [ognet](https://stanfordmlgroup.github.io/projects/ognet/) -> a Global Oil and Gas Infrastructure Database using Deep Learning on Remotely Sensed Imagery

  4.10.8. [RSOD-Dataset](https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-) -> dataset for object detection in PASCAL VOC format. Aircraft, playgrounds, overpasses & oiltanks. Used in the 2022 [paper](https://link.springer.com/article/10.1007/s00500-022-07106-8): Improved YOLOv5 network method for remote sensing image-based ground objects recognition

  4.10.9. [oil_storage-detector](https://github.com/TheodorEmanuelsson/oil_storage-detector) -> using yolov5 and the Airbus Oil Storage Detection dataset

  4.10.10. [oil_well_detector](https://github.com/dzubke/oil_well_detector) -> detect oil wells in the Bakken oil field based on satellite imagery

  4.10.11. [OGST](https://data.mendeley.com/datasets/bkxj8z84m9/3) -> Oil and Gas Tank Dataset

  4.10.12. [AContrarioTankDetection](https://github.com/anttad/AContrarioTankDetection) -> code for 2020 paper: Oil Tank Detection in Satellite Images via a Contrario Clustering

  4.10.13. [SubpixelCircleDetection](https://github.com/anttad/SubpixelCircleDetection) -> code for 2020 [paper](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/901/2020/): CIRCULAR-SHAPED OBJECT DETECTION IN LOW RESOLUTION SATELLITE IMAGES

  4.10.14. [Oil Storage Detection on Airbus Imagery with YOLOX](https://medium.com/artificialis/oil-storage-detection-on-airbus-imagery-with-yolox-9e38eb6f7e62) -> uses the Kaggle Airbus Oil Storage Detection dataset

### 4.11. Object detection - Animals

A variety of techniques can be used to count animals, including object detection and instance segmentation. For convenience they are all listed here:

  4.11.1. [cownter_strike](https://github.com/IssamLaradji/cownter_strike) -> counting cows, located with point-annotations, two models: CSRNet (a density-based method) & LCFCN (a detection-based method)

  4.11.2. [elephant_detection](https://github.com/akharina/elephant_detection) -> Using Keras-Retinanet to detect elephants from aerial images

  4.11.3. [CNN-Mosquito-Detection](https://github.com/sriramelango/CNN-Mosquito-Detection) -> determining the locations of potentially dangerous breeding grounds, compared YOLOv4, YOLOR & YOLOv5

  4.11.4. [Borowicz_etal_Spacewhale](https://github.com/lynch-lab/Borowicz_etal_Spacewhale) -> locate whales using ResNet

  4.11.5. [walrus-detection-and-count](https://github.com/sweetlhare/walrus-detection-and-count) -> uses Mask R-CNN instance segmentation

  4.11.6. [MarineMammalsDetection](https://github.com/Pangoraw/MarineMammalsDetection) -> Weakly Supervised Detection of Marine Animals in High Resolution Aerial Images

  4.11.7. [Audubon_F21](https://github.com/RiceD2KLab/Audubon_F21) -> code for 2022 [paper](https://arxiv.org/abs/2210.04868): Deep object detection for waterbird monitoring using aerial imagery

### 4.12. Object detection - Miscellaneous

  4.12.1. [Object detection on Satellite Imagery using RetinaNet](https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5) -> using the Kaggle Swimming Pool and Car Detection dataset `BEGINNER`

  4.12.2. [Tackling the Small Object Problem in Object Detection](https://blog.roboflow.com/tackling-the-small-object-problem-in-object-detection) `BEGINNER`

  4.12.3. [Object Detection and Image Segmentation with Deep Learning on Earth Observation Data: A Review](https://www.mdpi.com/2072-4292/12/10/1667)

  4.12.4. [awesome-aerial-object-detection bu murari023](https://github.com/murari023/awesome-aerial-object-detection), [another by visionxiang](https://github.com/visionxiang/awesome-object-detection-in-aerial-images) and [awesome-tiny-object-detection](https://github.com/kuanhungchen/awesome-tiny-object-detection) list many relevant papers

  4.12.5. [Object Detection Accuracy as a Function of Image Resolution](https://medium.com/the-downlinq/the-satellite-utility-manifold-object-detection-accuracy-as-a-function-of-image-resolution-ebb982310e8c) -> Medium article using COWC dataset, performance rapidly degrades below 30cm imagery

  4.12.6. [Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN)](https://github.com/avanetten/simrdwn) -> combines some of the leading object detection algorithms into a unified framework designed to detect objects both large and small in overhead imagery. Train models and test on arbitrary image sizes with YOLO (versions 2 and 3), Faster R-CNN, SSD, or R-FCN.

  4.12.7. [YOLTv4](https://github.com/avanetten/yoltv4) -> YOLTv4 is designed to detect objects in aerial or satellite imagery in arbitrarily large images that far exceed the ~600×600 pixel size typically ingested by deep learning object detection frameworks. Read [Announcing YOLTv4: Improved Satellite Imagery Object Detection](https://towardsdatascience.com/announcing-yoltv4-improved-satellite-imagery-object-detection-f5091e913fad)

  4.12.8. [Tensorflow Benchmarks for Object Detection in Aerial Images](https://github.com/yangxue0827/RotationDetection) -> tensorflow-based codebase created to build benchmarks for object detection in aerial images

  4.12.9. [Pytorch Benchmarks for Object Detection in Aerial Images](https://github.com/dingjiansw101/AerialDetection) -> pytorch-based codebase created to build benchmarks for object detection in aerial images using mmdetection

  4.12.10. [ASPDNet](https://github.com/liuqingjie/ASPDNet) -> Counting dense objects in remote sensing images, [arxiv paper](https://arxiv.org/abs/2002.05928)

  4.12.11. [xview-yolov3](https://github.com/ultralytics/xview-yolov3) -> xView 2018 Object Detection Challenge: YOLOv3 Training and Inference

  4.12.12. [Faster RCNN for xView satellite data challenge](https://github.com/samirsen/small-object-detection)

  4.12.13. [How to detect small objects in (very) large images](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98) -> A practical guide to using Slicing-Aided Hyper Inference (SAHI) for performing inference on the DOTAv1.0 object detection dataset using the mmdetection framework

  4.12.14. [Object Detection Satellite Imagery Multi-vehicles Dataset (SIMD)](https://github.com/asimniazi63/Object-Detection-on-Satellite-Images) -> RetinaNet,Yolov3 and Faster RCNN for multi object detection on satellite images dataset

  4.12.15. [SNIPER/AutoFocus](https://github.com/mahyarnajibi/SNIPER) -> an efficient multi-scale object detection training/inference algorithm

  4.12.16. [marine_debris_ML](https://github.com/NASA-IMPACT/marine_debris_ML) -> Marine debris detection, uses 3-meter imagery product called Planetscope with bands in the red, green, blue, and near-infrared. Uses Tensorflow Object Detection API with pre-trained resnet 101

  4.12.17. [pool-detection-from-aerial-imagery](https://towardsdatascience.com/pool-detection-from-aerial-imagery-f5b76d0a6093) -> Use Icevision and Detectron2 to detect swimming pools from aerial imagery

  4.12.18. [Electric-Pylon-Detection-in-RSI](https://github.com/qsjxyz/Electric-Pylon-Detection-in-RSI) -> a dataset which contains 1500 remote sensing images of electric pylons used to train ten deep learning models

  4.12.19. [Synthesizing Robustness YOLTv4 Results Part 2: Dataset Size Requirements and Geographic Insights](https://www.iqt.org/synthesizing-robustness-yoltv4-results-part-2-dataset-size-requirements-and-geographic-insights/) -> quantify how much harder rare objects are to localize

  4.12.20. [IS-Count](https://github.com/sustainlab-group/IS-Count) -> IS-Count is a sampling-based and learnable method for estimating the total object count in a region

  4.12.21. [Object Detection On Aerial Imagery Using RetinaNet](https://towardsdatascience.com/object-detection-on-aerial-imagery-using-retinanet-626130ba2203)

  4.12.22. [Clustered-Object-Detection-in-Aerial-Image](https://github.com/fyangneil/Clustered-Object-Detection-in-Aerial-Image)

  4.12.23. [yolov5s_for_satellite_imagery](https://github.com/KevinMuyaoGuo/yolov5s_for_satellite_imagery) -> yolov5s applied to the DOTA dataset

  4.12.24. [RetinaNet-PyTorch](https://github.com/HsLOL/RetinaNet-PyTorch) -> RetinaNet implementation on remote sensing ship dataset (SSDD)

  4.12.25. [Detecting-Cyclone-Centers-Custom-YOLOv3](https://github.com/ShubhayanS/Detecting-Cyclone-Centers-Custom-YOLOv3) -> tropical cyclones (TCs) are intense warm-corded cyclonic vortices, developed from low-pressure systems over the tropical oceans and driven by complex air-sea interaction

  4.12.26. [Object-Detection-YoloV3-RetinaNet-FasterRCNN](https://github.com/bostankhan6/Object-Detection-YoloV3-RetinaNet-FasterRCNN) -> trained on a private datset

  4.12.27. [Google-earth-Object-Recognition](https://github.com/InnovAIco/Google-earth-Object-Recognition) -> Code for training and evaluating on Dior Dataset (Google Earth Images) using RetinaNet and YOLOV5

  4.12.28. [HIECTOR: Hierarchical object detector at scale](https://medium.com/sentinel-hub/hiector-hierarchical-object-detector-at-scale-5a61753b51a3) -> HIECTOR facilitates multiple satellite data collections of increasingly detailed spatial resolution for a cost-efficient and accurate object detection over large areas. [Code on Github](https://github.com/sentinel-hub/hiector)

  4.12.29. [Detection of Multiclass Objects in Optical Remote Sensing Images](https://github.com/WenchaoliuMUC/Detection-of-Multiclass-Objects-in-Optical-Remote-Sensing-Images) -> code for 2018 paper: Detection of Multiclass Objects in Optical Remote Sensing Images

  4.12.30. [SB-MSN](https://github.com/weihancug/Sampling-Balance_Multi-stage_Network) -> Sampling-Balance based Multi-stage Network (SB-MSN) for aerial image object detection. Code for 2021 paper: Improving Training Instance Quality in Aerial Image Object Detection With a Sampling-Balance-Based Multistage Network

  4.12.31. [yoltv5](https://github.com/avanetten/yoltv5) -> detects objects in arbitrarily large aerial or satellite images that far exceed the ~600×600 pixel size typically ingested by deep learning object detection frameworks. Uses YOLOv5 & pytorch

  4.12.32. [AIR](https://github.com/Accenture/AIR) -> A deep learning object detector framework written in Python for supporting Land Search and Rescue Missions

  4.12.33. [dior_detect](https://github.com/hm-better/dior_detect) -> benchmarks for object detection on DIOR dataset

  4.12.34. [Panchromatic to Multispectral: Object Detection Performance as a Function of Imaging Bands](https://medium.com/the-downlinq/panchromatic-to-multispectral-object-detection-performance-as-a-function-of-imaging-bands-51ecaaa3dc56) -> Medium article, concludes that more bands are not always beneficial, but likely varies by use case

  4.12.35. [OPLD-Pytorch](https://github.com/yf19970118/OPLD-Pytorch) -> code for 2020 paper: Learning Point-Guided Localization for Detection in Remote Sensing Images

  4.12.36. [F3Net](https://github.com/yxhnjust/F3Net) -> code for 2020 [paper](https://www.mdpi.com/2072-4292/12/24/4027): Feature Fusion and Filtration Network for Object Detection in Optical Remote Sensing Images

  4.12.37. [GLNet](https://github.com/Zhu1Teng/GLNet) -> code for 2021 paper: Global to Local: Clip-LSTM-Based Object Detection From Remote Sensing Images

  4.12.38. [SRAF-Net](https://github.com/Complicateddd/SRAF-Net) -> code for 2021 paper: A Scene-Relevant Anchor-Free Object Detection Network in Remote Sensing Images

  4.12.39. [object_detection_in_remote_sensing_images](https://github.com/EEexplorer001/object_detection_in_remote_sensing_images) -> using CNN and attention mechanism

  4.12.40. [SHAPObjectDetection](https://github.com/hiroki-kawauchi/SHAPObjectDetection) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/9/1970): SHAP-Based Interpretable Object Detection Method for Satellite Imagery

  4.12.41. [NWD](https://github.com/jwwangchn/NWD) -> code for 2021 [paper](https://arxiv.org/abs/2110.13389): A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. Uses AI-TOD dataset

  4.12.42. [MSFC-Net](https://github.com/ZhAnGToNG1/MSFC-Net) -> code for 2021 paper: Multiscale Semantic Fusion-Guided Fractal Convolutional Object Detection Network for Optical Remote Sensing Imagery

  4.12.43. [LO-Det](https://github.com/Shank2358/LO-Det) -> code for the 2021 paper: LO-Det: Lightweight Oriented Object Detection in Remote Sensing Images

  4.12.44. [R2IPoints](https://github.com/shnew/R2IPoints) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9770816): R²IPoints: Pursuing Rotation-Insensitive Point Representation for Aerial Object Detection

  4.12.45. [Object-Detection](https://github.com/xiaojs18/Object-Detection) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/16/3969): Multi-Scale Object Detection with the Pixel Attention Mechanism in a Complex Background

  4.12.46. [mmdet-rfla](https://github.com/Chasel-Tsui/mmdet-rfla) -> code for 2022 [paper](https://arxiv.org/abs/2208.08738): RFLA: Gaussian Receptive based Label Assignment for Tiny Object Detection

  4.12.47. [Interactive-Multi-Class-Tiny-Object-Detection](https://github.com/ChungYi347/Interactive-Multi-Class-Tiny-Object-Detection) -> code for 2022 [paper](https://arxiv.org/abs/2203.15266): Interactive Multi-Class Tiny-Object Detection

  4.12.48. [small-object-detection-benchmark](https://github.com/fcakyon/small-object-detection-benchmark) -> code for ICIP 2022 [paper](https://arxiv.org/abs/2202.06934): Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection

  4.12.49. [OD-Satellite-iSAID](https://github.com/muzairkhattak/OD-Satellite-iSAID) -> Object Detection in Aerial Images: A Case Study on Performance Improvement using iSAID

  4.12.50. [Large-Selective-Kernel-Network](https://github.com/zcablii/Large-Selective-Kernel-Network) -> code for 2023 paper: Large Selective Kernel Network for Remote Sensing Object Detection

  4.12.51. [Satellite_Imagery_Detection_YOLOV7](https://github.com/Radhika-Keni/Satellite_Imagery_Detection_YOLOV7) -> YOLOV7 applied to xView1 Dataset

  4.12.52. [FSANet](https://github.com/Lausen-Ng/FSANet) -> code for 2022 paper: FSANet: Feature-and-Spatial-Aligned Network for Tiny Object Detection in Remote Sensing Images

  4.12.53. [OAN](https://github.com/Ranchosky/OAN) code for paper: Fewer is More: Efficient Object Detection in Large Aerial Images, based on MMdetection

  4.12.54. [DOTA-C](https://github.com/hehaodong530/DOTA-C) -> evaluating the robustness of object detection models to 19 types of image quality degradation

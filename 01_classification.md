# 1. Classification

<p align="center">
  <img src="images/merced.png" width="600">
  <br>
  <b>The UC merced dataset is a well known classification dataset.</b>
</p>

Classification is a fundamental task in remote sensing data analysis, where the goal is to assign a semantic label to each image, such as 'urban', 'forest', 'agricultural land', etc. The process of assigning labels to an image is known as image-level classification. However, in some cases, a single image might contain multiple different land cover types, such as a forest with a river running through it, or a city with both residential and commercial areas. In these cases, image-level classification becomes more complex and involves assigning multiple labels to a single image. This can be accomplished using a combination of feature extraction and machine learning algorithms to accurately identify the different land cover types. It is important to note that image-level classification should not be confused with pixel-level classification, also known as semantic segmentation. While image-level classification assigns a single label to an entire image, semantic segmentation assigns a label to each individual pixel in an image, resulting in a highly detailed and accurate representation of the land cover types in an image. Read [A brief introduction to satellite image classification with neural networks](https://medium.com/@robmarkcole/a-brief-introduction-to-satellite-image-classification-with-neural-networks-3ce28be15683)

   1.1. Land classification on Sentinel 2 data using a [simple sklearn cluster algorithm](https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/Clustering_KMeans-Sentinel2.ipynb) or [deep learning CNN](https://towardsdatascience.com/land-use-land-cover-classification-with-deep-learning-9a5041095ddb) `BEGINNER`

   1.2. Land Use Classification on Merced dataset using CNN [in Keras](https://github.com/tavgreen/landuse_classification)
  or [fastai](https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3). Also checkout [Multi-label Land Cover Classification](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d) using the redesigned multi-label Merced dataset with 17 land cover classes `BEGINNER`

   1.3. [Multi-Label Classification of Satellite Photos of the Amazon Rainforest using keras](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/) or [FastAI](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95) `BEGINNER`

   1.4. [EuroSat-Satellite-CNN-and-ResNet](https://github.com/Rumeysakeskin/EuroSat-Satellite-CNN-and-ResNet) -> Classifying custom image datasets by creating Convolutional Neural Networks and Residual Networks from scratch with PyTorch `BEGINNER`

   1.5. [Detecting Informal Settlements from Satellite Imagery using fine-tuning of ResNet-50 classifier](https://blog.goodaudience.com/detecting-informal-settlements-using-satellite-imagery-and-convolutional-neural-networks-d571a819bf44) with [repo](https://github.com/dymaxionlabs/ap-latam)

   1.6.  [Land-Cover-Classification-using-Sentinel-2-Dataset](https://github.com/raoofnaushad/Land-Cover-Classification-using-Sentinel-2-Dataset) -> [well written Medium article](https://raoofnaushad7.medium.com/applying-deep-learning-on-satellite-imagery-classification-5f2588b932c1) accompanying this repo but using the EuroSAT dataset

   1.7. [Land Cover Classification of Satellite Imagery using Convolutional Neural Networks](https://towardsdatascience.com/land-cover-classification-of-satellite-imagery-using-convolutional-neural-networks-91b5bb7fe808) using Keras and a multi spectral dataset captured over vineyard fields of Salinas Valley, California

   1.8. [Detecting deforestation from satellite images](https://towardsdatascience.com/detecting-deforestation-from-satellite-images-7aa6dfbd9f61) -> using FastAI and ResNet50, with repo [fsdl_deforestation_detection](https://github.com/karthikraja95/fsdl_deforestation_detection)

   1.9. [Neural Network for Satellite Data Classification Using Tensorflow in Python](https://towardsdatascience.com/neural-network-for-satellite-data-classification-using-tensorflow-in-python-a13bcf38f3e1) -> A step-by-step guide for Landsat 5 multispectral data classification for binary built-up/non-built-up class prediction, with [repo](https://github.com/PratyushTripathy/Landsat-Classification-Using-Neural-Network)

   1.10. [Slums mapping from pretrained CNN network](https://github.com/deepankverma/slums_detection) on VHR (Pleiades: 0.5m) and MR (Sentinel: 10m) imagery

   1.11. [Comparing urban environments using satellite imagery and convolutional neural networks](https://github.com/adrianalbert/urban-environments) -> includes interesting study of the image embedding features extracted for each image on the Urban Atlas dataset. Accompanying [paper](https://www.researchgate.net/publication/315882788_Using_convolutional_networks_and_satellite_imagery_to_identify_patterns_in_urban_environments_at_a_large_scale)

   1.12. [RSI-CB](https://github.com/lehaifeng/RSI-CB) -> A Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data. See also [Remote-sensing-image-classification](https://github.com/aashishrai3799/Remote-sensing-image-classification)

   1.13. [NAIP_PoolDetection](https://github.com/annaptasznik/NAIP_PoolDetection) -> modelled as an object recognition problem, a CNN is used to identify images as being swimming pools or something else - specifically a street, rooftop, or lawn

   1.14. [Land Use and Land Cover Classification using a ResNet Deep Learning Architecture](https://www.luigiselmi.eu/eo/lulc-classification-deeplearning.html) -> uses fastai and the EuroSAT dataset

   1.15. [Vision Transformers Use Case: Satellite Image Classification without CNNs](https://medium.com/nerd-for-tech/vision-transformers-use-case-satellite-image-classification-without-cnns-2c4dbeb06f87)

   1.16. [WaterNet](https://github.com/treigerm/WaterNet) -> a CNN that identifies water in satellite images

   1.17. [Road-Network-Classification](https://github.com/ualsg/Road-Network-Classification) -> Road network classification model using ResNet-34, road classes organic, gridiron, radial and no pattern

   1.18. [Scaling AI to map every school on the planet](https://developmentseed.org/blog/2021-03-18-ai-enabling-school-mapping)

   1.19.  [Landsat classification CNN tutorial](https://towardsdatascience.com/is-cnn-equally-shiny-on-mid-resolution-satellite-data-9e24e68f0c08) with [repo](https://github.com/PratyushTripathy/Landsat-Classification-Using-Convolution-Neural-Network)

   1.20. [satellite-crosswalk-classification](https://github.com/rodrigoberriel/satellite-crosswalk-classification)

   1.21. [Understanding the Amazon Rainforest with Multi-Label Classification + VGG-19, Inceptionv3, AlexNet & Transfer Learning](https://towardsdatascience.com/understanding-the-amazon-rainforest-with-multi-label-classification-vgg-19-inceptionv3-5084544fb655)

   1.22. [Implementation of the 3D-CNN model for land cover classification](https://medium.com/geekculture/remote-sensing-deep-learning-for-land-cover-classification-of-satellite-imagery-using-python-6a7b4c4f570f) -> uses the Sundarbans dataset, with [repo](https://github.com/syamkakarla98/Satellite_Imagery_Analysis). Also read [Land cover classification of Sundarbans satellite imagery using K-Nearest Neighbor(K-NNC), Support Vector Machine (SVM), and Gradient Boosting classification algorithms](https://towardsdatascience.com/land-cover-classification-in-satellite-imagery-using-python-ae39dbf2929) which is by the same author and shares the repo

   1.23. [SSTN](https://github.com/zilongzhong/SSTN) -> PyTorch Implementation of SSTNs for hyperspectral image classifications from the IEEE T-GRS paper "Spectral-Spatial Transformer Network for Hyperspectral Image Classification: A FAS Framework." Demonstrates a novel spectral-spatial transformer network (SSTN), which consists of spatial attention and spectral association modules, to overcome the constraints of convolution kernels

   1.24. [SatellitePollutionCNN](https://github.com/arnavbansal1/SatellitePollutionCNN) -> A novel algorithm to predict air pollution levels with state-of-art accuracy using deep learning and GoogleMaps satellite images

   1.25. [PropertyClassification](https://github.com/Sardhendu/PropertyClassification) -> Classifying the type of property given Real Estate, satellite and Street view Images

   1.26. [remote-sense-quickstart](https://github.com/CarryHJR/remote-sense-quickstart) -> classification on a number of datasets, including with attention visualization

   1.27. [Satellite image classification using multiple machine learning algorithms](https://github.com/tanmay-delhikar/satellite-image-analysis-ml)

   1.28. [satsense](https://github.com/DynaSlum/satsense) -> a Python library for land use/cover classification using classical features including HoG & NDVI

   1.29. [PyTorch_UCMerced_LandUse](https://github.com/GeneralLi95/PyTorch_UCMerced_LandUse) -> simple pytorch implementation fine tuned on ResNet and basic augmentations

   1.30. [EuroSAT-image-classification](https://github.com/artemisart/EuroSAT-image-classification) -> simple pytorch implementation fine tuned on ResNet

   1.31. [landcover_classification](https://github.com/reidfalconer/landcover_classification) -> using fast.ai on EuroSAT

   1.32. [IGARSS2020_BWMS](https://github.com/jiankang1991/IGARSS2020_BWMS) -> Band-Wise Multi-Scale CNN Architecture for Remote Sensing Image Scene Classification with a novel CNN architecture for the feature embedding of high-dimensional RS images

   1.33. [image.classification.on.EuroSAT](https://github.com/canturan10/image.classification.on.EuroSAT) -> solution in pure pytorch

   1.34. [hurricane_damage](https://github.com/allankapoor/hurricane_damage) -> Post-hurricane structure damage assessment based on aerial imagery with CNN

   1.35. [openai-drivendata-challenge](https://github.com/buildwithcycy/openai-drivendata-challenge) -> Using deep learning to classify the building material of rooftops (aerial imagery from South America)

   1.36. [is-it-abandoned](https://github.com/zach-brown-18/is-it-abandoned) -> Can we tell if a house is abandoned based on aerial LIDAR imagery?

   1.37. [BoulderAreaDetector](https://github.com/pszemraj/BoulderAreaDetector) -> CNN to classify whether a satellite image shows an area would be a good rock climbing spot or not

   1.38. [ISPRS_S2FL](https://github.com/danfenghong/ISPRS_S2FL) -> code for paper: Multimodal Remote Sensing Benchmark Datasets for Land Cover Classification with A Shared and Specific Feature Learning Model. S2FL is capable of decomposing multimodal RS data into modality-shared and modality-specific components, enabling the information blending of multi-modalities more effectively
   1.39. [Brazilian-Coffee-Detection](https://github.com/MrSquidward/Brazilian-Coffee-Detection) -> uses Keras with public dataset

   1.40. [tf-crash-severity](https://github.com/SoySauceNZ/tf-crash-severity) -> predict the crash severity for given road features contained within satellite images

   1.41. [ensemble_LCLU](https://github.com/burakekim/ensemble_LCLU) -> code for 2021 [paper](https://www.tandfonline.com/doi/full/10.1080/17538947.2021.1980125): Deep neural network ensembles for remote sensing land cover and land use classification

   1.42. [cerraNet](https://github.com/MirandaMat/cerraNet-v2) -> contextually classify the types of use and coverage in the Brazilian Cerrado

   1.43. [Urban-Analysis-Using-Satellite-Imagery](https://github.com/mominali12/Urban-Analysis-Using-Satellite-Imagery) -> classify urban area as planned or unplanned using a combination of segmentation and classification

   1.44. [ChipClassification](https://github.com/yurithefury/ChipClassification) -> code for 2019 [paper](https://www.sciencedirect.com/science/article/pii/S0924271619302023): Deep learning for multi-modal classification of cloud, shadow and land cover scenes in PlanetScope and Sentinel-2 imagery

   1.45. [DeeplearningClassficationLandsat-tImages](https://github.com/VinayarajPoliyapram/DeeplearningClassficationLandsat-tImages) -> Water/Ice/Land Classification Using Large-Scale Medium Resolution Landsat Satellite Images

   1.46. [wildfire-detection-from-satellite-images-ml](https://github.com/shrey24/wildfire-detection-from-satellite-images-ml) -> detect whether an image contains a wildfire, with example flask web app

   1.47. [mining-discovery-with-deep-learning](https://github.com/remis/mining-discovery-with-deep-learning) -> code for the 2020 paper: Mining and Tailings Dam Detection in Satellite Imagery Using Deep Learning

   1.48. [e-Farmerce-platform](https://github.com/efarmerce/e-Farmerce-platform) -> classify crop type

   1.49. [sentinel2-deep-learning](https://github.com/d-smit/sentinel2-deep-learning) -> Novel Training Methodologies for Land Classification of Sentinel-2 Imagery

   1.50. [RSSC-transfer](https://github.com/risojevicv/RSSC-transfer) -> code for 2021 [paper](https://arxiv.org/abs/2111.03690): The Role of Pre-Training in High-Resolution Remote Sensing Scene Classification

   1.51. [Classifying Geo-Referenced Photos and Satellite Images for Supporting Terrain Classification](https://github.com/jorgemspereira/Classifying-Geo-Referenced-Photos) -> detect floods

   1.52. [Pay-More-Attention](https://github.com/williamzhao95/Pay-More-Attention) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9157951): Remote Sensing Image Scene Classification Based on an Enhanced Attention Module

   1.53. [Remote-Sensing-Image-Classification-via-Improved-Cross-Entropy-Loss-and-Transfer-Learning-Strategy](https://github.com/AliBahri94/Remote-Sensing-Image-Classification-via-Improved-Cross-Entropy-Loss-and-Transfer-Learning-Strategy) -> code for 2019 [paper](https://ieeexplore.ieee.org/abstract/document/8844264): Remote Sensing Image Classification via Improved Cross-Entropy Loss and Transfer Learning Strategy Based on Deep Convolutional Neural Networks

   1.54. [DenseNet40-for-HRRSISC](https://github.com/BiQiWHU/DenseNet40-for-HRRSISC) -> DenseNet40 for remote sensing image scene classification, uses UC Merced Dataset

   1.55. [SKAL](https://github.com/hw2hwei/SKAL) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9298485): Looking Closer at the Scene: Multiscale Representation Learning for Remote Sensing Image Scene Classification

   1.56. [potsdam-tensorflow-practice](https://github.com/medicinely/potsdam-tensorflow-practice) -> image classification of Potsdam dataset using tensorflow

   1.57. [SAFF](https://github.com/zh-hike/SAFF) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/8982033): Self-Attention-Based Deep Feature Fusion for Remote Sensing Scene Classification

   1.58. [GLNET](https://github.com/wuchangsheng951/GLNET) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9607791): Convolutional Neural Networks Based Remote Sensing Scene Classification under Clear and Cloudy Environments

   1.59. [Remote-sensing-image-classification](https://github.com/hiteshK03/Remote-sensing-image-classification) -> transfer learning using pytorch to classify remote sensing data into three classes: aircrafts, ships, none

   1.60. [remote_sensing_pretrained_models](https://github.com/lsh1994/remote_sensing_pretrained_models) -> as an alternative to fine tuning on models pretrained on ImageNet, here some CNN are pretrained on the RSD46-WHU & AID datasets

   1.61. [CNN_AircraftDetection](https://github.com/UKMIITB/CNN_AircraftDetection) -> CNN for aircraft detection in satellite images using keras

   1.62. [OBIC-GCN](https://github.com/CVEO/OBIC-GCN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9411513): Object-based Classification Framework of Remote Sensing Images with Graph Convolutional Networks

   1.63. [aitlas-arena](https://github.com/biasvariancelabs/aitlas-arena) -> An open-source benchmark framework for evaluating state-of-the-art deep learning approaches for image classification in Earth Observation (EO)

   1.64. [droughtwatch](https://github.com/wandb/droughtwatch) -> code for 2020 [paper](https://arxiv.org/abs/2004.04081): Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya

   1.65. [JSTARS_2020_DPN-HRA](https://github.com/B-Xi/JSTARS_2020_DPN-HRA) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9126161): Deep Prototypical Networks With Hybrid Residual Attention for Hyperspectral Image Classification

   1.66. [SIGNA](https://github.com/kyle-one/SIGNA) -> code for 2022 [paper](https://arxiv.org/abs/2208.02613): Semantic Interleaving Global Channel Attention for Multilabel Remote Sensing Image Classification

   1.67. [Satellite Image Classification](https://github.com/rocketmlhq/rmldnn/tree/main/tutorials/satellite_image_classification) using rmldnn and Sentinel 2 data

   1.68. [PBDL](https://github.com/Usman1021/PBDL) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/23/5913): Patch-Based Discriminative Learning for Remote Sensing Scene Classification

   1.69. [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) -> identify fire and other emergencies from a drone

   1.70. [satellite-deforestation](https://github.com/drewhibbard/satellite-deforestation) -> Using Satellite Imagery to Identify the Leading Indicators of Deforestation, applied to the Kaggle Challenge Understanding the Amazon from Space

   1.71. [RSMLC](https://github.com/marjanstoimchev/RSMLC) -> code for 2023 [paper](https://www.mdpi.com/2072-4292/15/2/538): Deep Network Architectures as Feature Extractors for Multi-Label Classification of Remote Sensing Images

   1.72. [FireRisk](https://github.com/CharmonyShen/FireRisk) -> A Remote Sensing Dataset for Fire Risk Assessment with Benchmarks Using Supervised and Self-supervised Learning

   1.73. [flood_susceptibility_mapping](https://github.com/omarseleem92/flood_susceptibility_mapping) -> Towards urban flood susceptibility mapping using data-driven models in Berlin, Germany

   1.74. [tick-tick-bloom](https://github.com/drivendataorg/tick-tick-bloom) -> Winners of the Tick Tick Bloom: Harmful Algal Bloom Detection Challenge. Task was to predict severity of algae bloom, winners used decision trees

   1.75. [Estimating coal power plant operation from satellite images with computer vision](https://transitionzero.medium.com/estimating-coal-power-plant-operation-from-satellite-images-with-computer-vision-b966af56919e) -> use Sentinel 2 data to identify if a coal power plant is on or off, with dataset and repo

   1.76. [Building-detection-and-roof-type-recognition](https://github.com/loosgagnet/Building-detection-and-roof-type-recognition) -> A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image

   1.77 [Performance Comparison of Multispectral Channels for Land Use Classification](https://github.com/tejasri19/EuroSAT_data_analysis) -> Implemented ResNet-50, ResNet-101, ResNet-152, Vision Transformer on RGB and multispectral versions of EuroSAT dataset.

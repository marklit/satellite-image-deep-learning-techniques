# 2. Segmentation

<p align="center">
  <img src="images/segmentation.png" width="500">
  <br>
  <b>(left) a satellite image and (right) the semantic classes in the image.</b>
</p>

Image segmentation is a crucial step in image analysis and computer vision, with the goal of dividing an image into semantically meaningful segments or regions. The process of image segmentation assigns a class label to each pixel in an image, effectively transforming an image from a 2D grid of pixels into a 2D grid of pixels with assigned class labels. One common application of image segmentation is road or building segmentation, where the goal is to identify and separate roads and buildings from other features within an image. To accomplish this task, single class models are often trained to differentiate between roads and background, or buildings and background. These models are designed to recognize specific features, such as color, texture, and shape, that are characteristic of roads or buildings, and use this information to assign class labels to the pixels in an image. Another common application of image segmentation is land use or crop type classification, where the goal is to identify and map different land cover types within an image. In this case, multi-class models are typically used to recognize and differentiate between multiple classes within an image, such as forests, urban areas, and agricultural land. These models are capable of recognizing complex relationships between different land cover types, allowing for a more comprehensive understanding of the image content. Read [A brief introduction to satellite image segmentation with neural networks](https://medium.com/@robmarkcole/a-brief-introduction-to-satellite-image-segmentation-with-neural-networks-33ea732d5bce). **Note** that many articles which refer to 'hyperspectral land classification' are often actually describing semantic segmentation. [Image source](https://towardsdatascience.com/semantic-segmentation-of-aerial-imagery-using-u-net-in-python-552705238514)

### 2.1. Segmentation - Land use & land cover

  2.1.1. [U-Net for Semantic Segmentation on Unbalanced Aerial Imagery](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56) -> using the Dubai dataset `BEGINNER`

  2.1.2. [Semantic Segmentation of Dubai dataset Using a TensorFlow U-Net Model](https://towardsdatascience.com/semantic-segmentation-of-aerial-imagery-using-u-net-in-python-552705238514) `BEGINNER`

  2.1.3. [nga-deep-learning](https://github.com/jordancaraballo/nga-deep-learning) -> performs semantic segmentation on high resultion GeoTIF data using a modified U-Net & Keras, published by NASA researchers

  2.1.4. [Automatic Detection of Landfill Using Deep Learning](https://github.com/AnupamaRajkumar/LandfillDetection_SemanticSegmentation)

  2.1.5. [SpectralNET](https://github.com/tanmay-ty/SpectralNET) -> a 2D wavelet CNN for Hyperspectral Image Classification, uses Salinas Scene dataset & Keras

  2.1.6. [laika](https://github.com/datasciencecampus/laika) -> The goal of this repo is to research potential sources of satellite image data and to implement various algorithms for satellite image segmentation

  2.1.7. [PEARL](https://www.landcover.io/) -> a human-in-the-loop AI tool to drastically reduce the time required to produce an accurate Land Use/Land Cover (LULC) map, [blog post](http://devseed.com/blog/2021-05-17-pearl-ai-land-cover), uses Microsoft Planetary Computer and ML models run locally in the browser. Code for [backelnd](https://github.com/developmentseed/pearl-backend) and [frontend](https://github.com/developmentseed/pearl-frontend)

  2.1.8. [Land Cover Classification with U-Net](https://baratam-tarunkumar.medium.com/land-cover-classification-with-u-net-aa618ea64a1b) -> Satellite Image Multi-Class Semantic Segmentation Task with PyTorch Implementation of U-Net, uses DeepGlobe Land Cover Segmentation dataset, with [code](https://github.com/TarunKumar1995-glitch/land_cover_classification_unet)

  2.1.9. [Multi-class semantic segmentation of satellite images using U-Net](https://github.com/rogerxujiang/dstl_unet) using DSTL dataset, tensorflow 1 & python 2.7. Accompanying [article](https://towardsdatascience.com/dstl-satellite-imagery-contest-on-kaggle-2f3ef7b8ac40)

  2.1.10. [Codebase for multi class land cover classification with U-Net](https://github.com/jaeeolma/lulc_ml) accompanying a masters thesis, uses Keras

  2.1.11. [dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation) -> due to the small dataset, image augmentation was used

  2.1.12. [CDL-Segmentation](https://github.com/asimniazi63/CDL-Segmentation) -> code for the 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9441483): Deep Learning Based Land Cover and Crop Type Classification: A Comparative Study. Compares UNet, SegNet & DeepLabv3+

  2.1.13. [LoveDA](https://github.com/Junjue-Wang/LoveDA) -> code for the 2021 [paper](https://arxiv.org/abs/2110.08733): A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation

  2.1.14. [Satellite Imagery Semantic Segmentation with CNN](https://joshting.medium.com/satellite-imagery-segmentation-with-convolutional-neural-networks-f9254de3b907) -> 7 different segmentation classes, DeepGlobe Land Cover Classification Challenge dataset, with [repo](https://github.com/justjoshtings/satellite_image_segmentation)

  2.1.15. [Aerial Semantic Segmentation using U-Net Deep Learning Model](https://medium.com/@rehman.aimal/aerial-semantic-segmentation-using-u-net-deep-learning-model-3356a53c915f) medium article, with [repo](https://github.com/aimalrehman92/Multiclass-Semantic-Segmentation-with-U-NET)

  2.1.16. [UNet-Satellite-Image-Segmentation](https://github.com/YudeWang/UNet-Satellite-Image-Segmentation) -> A Tensorflow implentation of light UNet semantic segmentation framework

  2.1.17. [DeepGlobe Land Cover Classification Challenge solution](https://github.com/GeneralLi95/deepglobe_land_cover_classification_with_deeplabv3plus)

  2.1.18. [Semantic-segmentation-with-PyTorch-Satellite-Imagery](https://github.com/JenAlchimowicz/Semantic-segmentation-with-PyTorch-Satellite-Imagery) -> predict 25 classes on RGB imagery taken to assess the damage after Hurricane Harvey

  2.1.19. [Semantic Segmentation With Sentinel-2 Imagery](https://github.com/pavlo-seimskyi/semantic-segmentation-satellite-imagery) -> uses LandCoverNet dataset and fast.ai

  2.1.20. [CNN_Enhanced_GCN](https://github.com/qichaoliu/CNN_Enhanced_GCN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9268479): CNN-Enhanced Graph Convolutional Network With Pixel- and Superpixel-Level Feature Fusion for Hyperspectral Image Classification

  2.1.21. [LULCMapping-WV3images-CORINE-DLMethods](https://github.com/esertel/LULCMapping-WV3images-CORINE-DLMethods) -> Land Use and Land Cover Mapping Using Deep Learning Based Segmentation Approaches and VHR Worldview-3 Images

  2.1.22. [SOLC](https://github.com/yisun98/SOLC) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S0303243421003457): MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification. Uses [WHU-OPT-SAR-dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset)

  2.1.23. [MUnet-LUC](https://github.com/abhi170599/MUnet-LUC) -> Land Use with mUnet

  2.1.24. [land-cover](https://github.com/lucashu1/land-cover) -> code for 2021 [paper](https://arxiv.org/abs/2008.10351): Model Generalization in Deep Learning Applications for Land Cover Mapping

  2.1.25. [generalizablersc](https://github.com/dgominski/generalizablersc) -> code for 2022 paper: Cross-dataset Learning for Generalizable Land Use Scene Classification

  2.1.26. [Large-scale-Automatic-Identification-of-Urban-Vacant-Land](https://github.com/SkydustZ/Large-scale-Automatic-Identification-of-Urban-Vacant-Land) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0169204622000330): Large-scale automatic identification of urban vacant land using semantic segmentation of high-resolution remote sensing images

  2.1.27. [SSLTransformerRS](https://github.com/HSG-AIML/SSLTransformerRS) -> code for 2022 paper: Self-supervised Vision Transformers for Land-cover Segmentation and
  Classification

  2.1.28. [aerial-tile-segmentation](https://github.com/mrsebai/aerial-tile-segmentation) -> Large satellite image semantic segmentation into 6 classes using Tensorflow 2.0 and ISPRS benchmark dataset

  2.1.29. [LULCMapping-WV3images-CORINE-DLMethods](https://github.com/burakekim/LULCMapping-WV3images-CORINE-DLMethods) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/18/4558): Land Use and Land Cover Mapping Using Deep Learning Based Segmentation Approaches and VHR Worldview-3 Images

  2.1.30. [DCSA-Net](https://github.com/Julia90/DCSA-Net) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/19/4941): Dynamic Convolution Self-Attention Network for Land-Cover Classification in VHR Remote-Sensing Images

  2.1.31. [CHeGCN-CNN_enhanced_Heterogeneous_Graph](https://github.com/Liuzhizhiooo/CHeGCN-CNN_enhanced_Heterogeneous_Graph) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/19/5027): CNN-Enhanced Heterogeneous Graph Convolutional Network: Inferring Land Use from Land Cover with a Case Study of Park Segmentation

  2.1.32. [TCSVT_2022_DGSSC](https://github.com/B-Xi/TCSVT_2022_DGSSC) -> code for the 2022 [paper](https://ieeexplore.ieee.org/document/9924229): DGSSC: A Deep Generative Spectral-Spatial Classifier for Imbalanced Hyperspectral Imagery

  2.1.33. [DeepForest-Wetland-Paper](https://github.com/aj1365/DeepForest-Wetland-Paper) -> code for 2021 paper: Deep Forest classifier for wetland mapping using the combination of Sentinel-1 and Sentinel-2 data, GIScience & Remote Sensing

  2.1.34. [Wetland_UNet](https://github.com/conservation-innovation-center/Wetland_UNet) -> UNet models that can delineate wetlands using remote sensing data input including bands from Sentinel-2 LiDAR and geomorphons. By the Conservation Innovation Center of Chesapeake Conservancy and Defenders of Wildlife

  2.1.35. [DPA](https://github.com/x-ytong/DPA) -> DPA is an unsupervised domain adaptation (UDA) method applied to different satellite images for larg-scale land cover mapping.

  2.1.36. [dynamicworld](https://github.com/google/dynamicworld) -> code for 2022 [paper](https://www.nature.com/articles/s41597-022-01307-4): Dynamic World, Near real-time global 10 m land use land cover mapping

  2.1.37. [spada](https://github.com/links-ads/spada) -> code for 2023 [paper](https://arxiv.org/abs/2306.16252): Land Cover Segmentation with Sparse Annotations from Sentinel-2 Imagery

  2.1.38. [M3SPADA](https://github.com/ecapliez/M3SPADA) -> code for 2023 paper: Multi-Sensor Temporal Unsupervised Domain Adaptation for Land Cover Mapping with spatial pseudo labelling and adversarial learning

  2.1.39. [GLNet](https://github.com/VITA-Group/GLNet) -> code for 2019 paper: Collaborative Global-Local Networks for Memory-Efﬁcient Segmentation of Ultra-High Resolution Images

### 2.2. Segmentation - Vegetation, deforestation, crops & crop boundaries

Note that deforestation detection may be treated as a segmentation task or a change detection task

  2.2.1. [Сrор field boundary detection: approaches and main challenges](https://medium.com/geekculture/%D1%81r%D0%BE%D1%80-field-boundary-detection-approaches-and-main-challenges-46e37dd276bc) -> Medium article, covering historical and modern approaches `BEGINNER`

  2.2.2. [kenya-crop-mask](https://github.com/nasaharvest/kenya-crop-mask) -> Annual and in-season crop mapping in Kenya - LSTM classifier to classify pixels as containing crop or not, and a multi-spectral forecaster that provides a 12 month time series given a partial input. Dataset downloaded from GEE and pytorch lightning used for training `BEGINNER`

  2.2.3. [What’s growing there? Identify crops from multi-spectral remote sensing data (Sentinel 2)](https://towardsdatascience.com/whats-growing-there-a5618a2e6933) using eo-learn for data pre-processing, cloud detection, NDVI calculation, image augmentation & fastai

  2.2.4. [Tree species classification from from airborne LiDAR and hyperspectral data using 3D convolutional neural networks](https://github.com/jaeeolma/tree-detection-evo) accompanies research paper and uses fastai

  2.2.5. [crop-type-classification](https://medium.com/nerd-for-tech/crop-type-classification-cf5cc2593396) -> using Sentinel 1 & 2 data with a U-Net + LSTM, more features (i.e. bands) and higher resolution produced better results (article, no code)

  2.2.6. [Find sports fields using Mask R-CNN and overlay on open-street-map](https://github.com/jremillard/images-to-osm)

  2.2.7. [An LSTM to generate a crop mask for Togo](https://github.com/nasaharvest/togo-crop-mask)

  2.2.8. [DeepSatModels](https://github.com/michaeltrs/DeepSatModels) -> Code for paper "Context-self contrastive pretraining for crop type semantic segmentation"

  2.2.9. [farm-pin-crop-detection-challenge](https://github.com/simongrest/farm-pin-crop-detection-challenge) -> Using eo-learn and fastai to identify crops from multi-spectral remote sensing data

  2.2.10. [Detecting Agricultural Croplands from Sentinel-2 Satellite Imagery](https://medium.com/radiant-earth-insights/detecting-agricultural-croplands-from-sentinel-2-satellite-imagery-a025735d3bd8) -> We developed UNet-Agri, a benchmark machine learning model that classifies croplands using open-access Sentinel-2 imagery at 10m spatial resolution

  2.2.11. [DeepTreeAttention](https://github.com/weecology/DeepTreeAttention) -> Implementation of Hang et al. 2020 "Hyperspectral Image Classification with Attention Aided CNNs" for tree species prediction

  2.2.12. [Crop-Classification](https://github.com/bhavesh907/Crop-Classification) -> crop classification using multi temporal satellite images

  2.2.13. [ParcelDelineation](https://github.com/sustainlab-group/ParcelDelineation) -> using a French polygons dataset and unet in keras

  2.2.14. [crop-mask](https://github.com/nasaharvest/crop-mask) -> End-to-end workflow for generating high resolution cropland maps, uses GEE & LSTM model

  2.2.15. [DeepCropMapping](https://github.com/Lab-IDEAS/DeepCropMapping) -> A multi-temporal deep learning approach with improved spatial generalizability for dynamic corn and soybean mapping, uses LSTM

  2.2.16. [Segment Canopy Cover and Soil using NDVI and Rasterio](https://towardsdatascience.com/segment-satellite-imagery-using-ndvi-and-rasterio-6dcae02a044b)

  2.2.17. [Use KMeans clustering to segment satellite imagery by land cover/land use](https://towardsdatascience.com/segment-satellite-images-using-rasterio-and-scikit-learn-fc048f465874)

  2.2.18. [ResUnet-a](https://github.com/Akhilesh64/ResUnet-a) -> Implementation of the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data" in TensorFlow

  2.2.19. [DSD_paper_2020](https://github.com/JacobJeppesen/DSD_paper_2020) -> The code for the paper: Crop Type Classification based on Machine Learning with Multitemporal Sentinel-1 Data

  2.2.20. [MR-DNN](https://github.com/yasir2afaq/Multi-resolution-deep-neural-network) -> extract rice field from Landsat 8 satellite imagery

  2.2.21. [deep_learning_forest_monitoring](https://github.com/waldeland/deep_learning_forest_monitoring) -> Estimate vegetation height, code for paper: Forest mapping and monitoring of the African continent using Sentinel-2 data and deep learning

  2.2.22. [global-cropland-mapping](https://github.com/Charly-tian/global-cropland-mapping) -> global multi-temporal cropland mapping

  2.2.23. [U-Net for Semantic Segmentation of Soyabean Crop Fields with SAR images](https://joaootavionf007.medium.com/u-net-for-semantic-segmentation-of-soyabeans-crop-fields-with-sar-images-604232e49315)

  2.2.24. [UNet-RemoteSensing](https://github.com/aryanVijaywargia/UNet-RemoteSensing) -> uses 7 bands of Landsat and keras

  2.2.25. [Landuse_DL](https://github.com/yghlc/Landuse_DL) -> delineate landforms due to the thawing of ice-rich permafrost

  2.2.26. [canopy](https://github.com/jonathanventura/canopy) -> code for 2019 [paper](https://www.mdpi.com/2072-4292/11/19/2326): A Convolutional Neural Network Classifier Identifies Tree Species in Mixed-Conifer Forest from Hyperspectral Imagery

  2.2.27. [RandomForest-Classification](https://github.com/florianbeyer/RandomForest-Classification) -> script is for random forest classification of remote sensing multi-band images, used in 2019 [paper](https://www.tandfonline.com/doi/abs/10.1080/01431161.2019.1580825): Multisensor data to derive peatland vegetation communities using a fixed-wing unmanned aerial vehicle

  2.2.28. [forest_change_detection](https://github.com/QuantuMobileSoftware/forest_change_detection) -> forest change segmentation with time-dependent models, including Siamese, UNet-LSTM, UNet-diff, UNet3D models. Code for 2021 [paper](https://ieeexplore.ieee.org/document/9241044): Deep Learning for Regular Change Detection in Ukrainian Forest Ecosystem With Sentinel-2

  2.2.29. [cultionet](https://github.com/jgrss/cultionet) -> segmentation of cultivated land, built on PyTorch Geometric and PyTorch Lightning

  2.2.30. [sentinel-tree-cover](https://github.com/wri/sentinel-tree-cover) -> code for 2020 [paper](https://arxiv.org/abs/2005.08702): A global method to identify trees outside of closed-canopy forests with medium-resolution satellite imagery

  2.2.31. [crop-type-detection-ICLR-2020](https://github.com/RadiantMLHub/crop-type-detection-ICLR-2020) -> Winning Solutions from Crop Type Detection Competition at CV4A workshop, ICLR 2020

  2.2.32. [Crop identification using satellite imagery](https://write.agrevolution.in/crop-identification-using-satellite-imagery-introduction-83d79344f9ee) -> Medium article, introduction to crop identification

  2.2.33. [S4A-Models](https://github.com/Orion-AI-Lab/S4A-Models) -> Various experiments on the Sen4AgriNet dataset

  2.2.34. [attention-mechanism-unet](https://github.com/davej23/attention-mechanism-unet) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S0303243422000113): An attention-based U-Net for detecting deforestation within satellite sensor imagery

  2.2.35. [Cocoa_plantations_detection](https://github.com/antoine-spahr/Cocoa_plantations_detection) -> Detecting cocoa plantation in Ivory Coast using Sentinel-2 remote sensing data using KNN, SVM, Random Forest and MLP

  2.2.36. [SummerCrop_Deeplearning](https://github.com/AgriRS/SummerCrop_Deeplearning) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/20/5216): A Transferable Learning Classification Model and Carbon Sequestration Estimation of Crops in Farmland Ecosystem

  2.2.37. [DeepForest](https://deepforest.readthedocs.io/en/latest/index.html) is a python package for training and predicting individual tree crowns from airborne RGB imagery

  2.2.38. [Official repository for the "Identifying trees on satellite images" challenge from Omdena](https://github.com/cienciaydatos/ai-challenge-trees)

  2.2.39. [Counting-Trees-using-Satellite-Images](https://github.com/A2Amir/Counting-Trees-using-Satellite-Images) -> create an inventory of incoming and outgoing trees for an annual tree inspections, uses keras & semantic segmentation

  2.2.40. [2020 Nature paper - An unexpectedly large count of trees in the West African Sahara and Sahel](https://www.nature.com/articles/s41586-020-2824-5) -> tree detection framework based on U-Net & tensorflow 2 with code [here](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel/tree/v1.0.0)

  2.2.41. [TreeDetection](https://github.com/AmirNiaraki/TreeDetection) -> A color-based classifier to detect the trees in google image data along with tree visual localization and crown size calculations via OpenCV

  2.2.42. [PTDM](https://github.com/hr8yhtzb/PTDM) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/16/3902): Pomelo Tree Detection Method Based on Attention Mechanism and Cross-Layer Feature Fusion

  2.2.43. [urban-tree-detection](https://github.com/jonathanventura/urban-tree-detection) -> code for 2022 [paper](https://arxiv.org/abs/2208.10607): Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery. With [dataset](https://github.com/jonathanventura/urban-tree-detection-data)

  2.2.44. [BioMassters_baseline](https://github.com/fnands/BioMassters_baseline) -> a basic pytorch lightning baseline using a UNet for getting started with the [BioMassters challenge](https://www.drivendata.org/competitions/99/biomass-estimation/) (biomass estimation)

  2.2.45. [Biomassters winners](https://github.com/drivendataorg/the-biomassters) -> top 3 solutions

  2.2.46. [kbrodt biomassters solution](https://github.com/kbrodt/biomassters) -> 1st place solution

  2.2.47. [quqixun biomassters solution](https://github.com/quqixun/BioMassters)

  2.2.48. [biomass-estimation](https://github.com/azavea/biomass-estimation) -> from Azavea, applied to Sentinel 1 & 2

  2.2.49. [3DUNetGSFormer](https://github.com/aj1365/3DUNetGSFormer) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S1574954122003545): 3DUNetGSFormer: A deep learning pipeline for complex wetland mapping using generative adversarial networks and Swin transformer

  2.2.50. [SEANet_torch](https://github.com/long123524/SEANet_torch) -> code for 2023 paper: Using a semantic edge-aware multi-task neural network to delineate agricultural parcels from remote sensing images

  2.2.51. [arborizer](https://github.com/RaffiBienz/arborizer) -> Tree crowns segmentation and classification

  2.2.52. [ReUse](https://github.com/priamus-lab/ReUse) -> UNet to estimate carbon absorbed by forests, using Biomass & Sentinel-2 imagery. Code for [paper](https://www.mdpi.com/2313-433X/9/3/61): ReUse: REgressive Unet for Carbon Storage and Above-Ground Biomass Estimation

  2.2.53. [unet-sentinel](https://github.com/eliasqueirogavieira/unet-sentinel) -> UNet to handle Sentinel-1 SAR images to identify deforestation

  2.2.54. [MaskedSST](https://github.com/HSG-AIML/MaskedSST) -> code for 2023 paper: Masked Vision Transformers for Hyperspectral Image Classification

  2.2.55. [UNet-defmapping](https://github.com/bragagnololu/UNet-defmapping) -> master's thesis using UNet to map deforestation using Sentinel-2 Level 2A images, applied to Amazon and Atlantic Rainforest dataset

  2.2.56. [cvpr-multiearth-deforestation-segmentation](https://github.com/h2oai/cvpr-multiearth-deforestation-segmentation) -> multimodal Unet entry to the CVPR Multiearth 2023 deforestation challenge

  2.2.57. [supervised-wheat-classification-using-pytorchs-torchgeo](https://medium.com/@sulemanhamdani10/supervised-wheat-classification-using-pytorchs-torchgeo-combining-satellite-imagery-and-python-fc7f95c82e) -> Article: supervised wheat classification using torchgeo `BEGINNER`

  2.2.58. [TransUNetplus2](https://github.com/aj1365/TransUNetplus2) -> code for 2023 paper: TransU-Net++: Rethinking attention gated TransU-Net for deforestation mapping. Uses the Amazon and Atlantic forest dataset

  2.2.59. [A high-resolution canopy height model of the Earth](https://github.com/langnico/global-canopy-height-model#a-high-resolution-canopy-height-model-of-the-earth) -> code for 2022 paper: A high-resolution canopy height model of the Earth

### 2.3. Segmentation - Water, coastlines & floods

  2.3.1. [pytorch-waterbody-segmentation](https://github.com/gauthamk02/pytorch-waterbody-segmentation) -> UNET model trained on the Satellite Images of Water Bodies dataset from Kaggle. The model is deployed on Hugging Face Spaces `BEGINNER`

  2.3.2. [Flood Detection and Analysis using UNET with Resnet-34 as the back bone](https://github.com/orion29/Satellite-Image-Segmentation-for-Flood-Damage-Analysis) uses fastai `BEGINNER`

  2.3.3. [Automatic Flood Detection from Satellite Images Using Deep Learning](https://medium.com/@omercaliskan99/automatic-flood-detection-from-satellite-images-using-deep-learning-f14fafd369e0) `BEGINNER`

  2.3.4. [UNSOAT used fastai to train a Unet to perform semantic segmentation on satellite imageries to detect water](https://forums.fast.ai/t/unosat-used-fastai-ai-for-their-floodai-model-discussion-on-how-to-move-forward/78468) - [paper](https://www.mdpi.com/2072-4292/12/16/2532) + [notebook](https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb), accuracy 0.97, precision 0.91, recall 0.92

  2.3.5. [Semi-Supervised Classification and Segmentation on High Resolution Aerial Images - Solving the FloodNet problem](https://sahilkhose.medium.com/paper-presentation-e9bd0f3fb0bf)

  2.3.6. [Houston_flooding](https://github.com/Lichtphyz/Houston_flooding) -> labeling each pixel as either flooded or not using data from Hurricane Harvey. Dataset consisted of pre and post flood images, and a ground truth floodwater mask was created using unsupervised clustering (with DBScan) of image pixels with human cluster verification/adjustment

  2.3.7. [ml4floods](https://github.com/spaceml-org/ml4floods) -> An ecosystem of data, models and code pipelines to tackle flooding with ML

  2.3.8. [A comprehensive guide to getting started with the ETCI Flood Detection competition](https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a) -> using Sentinel1 SAR & pytorch

  2.3.9. [Map Floodwater of SAR Imagery with SageMaker](https://github.com/JayThibs/map-floodwater-sar-imagery-on-sagemaker) -> applied to Sentinel-1 dataset

  2.3.10. [1st place solution for STAC Overflow: Map Floodwater from Radar Imagery hosted by Microsoft AI for Earth](https://github.com/sweetlhare/STAC-Overflow) -> combines Unet with Catboostclassifier, taking their maxima, not the average

  2.3.11. [hydra-floods](https://github.com/Servir-Mekong/hydra-floods) -> an open source Python application for downloading, processing, and delivering surface water maps derived from remote sensing data

  2.3.12. [CoastSat](https://github.com/kvos/CoastSat) -> tool for mapping coastlines which has an extension [CoastSeg](https://github.com/dbuscombe-usgs/CoastSeg) using  segmentation models

  2.3.13. [Satellite_Flood_Segmentation_of_Harvey](https://github.com/morgan-tam/Satellite_Flood_Segmentation_of_Harvey) -> explores both deep learning and traditional kmeans

  2.3.14. [Flood Event Detection Utilizing Satellite Images](https://github.com/KonstantinosF/Flood-Detection---Satellite-Images)

  2.3.15. [ETCI-2021-Competition-on-Flood-Detection](https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection) -> Experiments on Flood Segmentation on Sentinel-1 SAR Imagery with Cyclical Pseudo Labeling and Noisy Student Training, with [arxiv paper](https://arxiv.org/abs/2107.08369)

  2.3.16. [FDSI](https://github.com/keillernogueira/FDSI) -> Flood Detection in Satellite Images - 2017 Multimedia Satellite Task

  2.3.17. [deepwatermap](https://github.com/isikdogan/deepwatermap) -> a deep model that segments water on multispectral images

  2.3.18. [rivamap](https://github.com/isikdogan/rivamap) -> an automated river analysis and mapping engine

  2.3.19. [deep-water](https://github.com/maxbeber/deep-water) -> track changes in water level

  2.3.20. [WatNet](https://github.com/xinluo2018/WatNet) -> A deep ConvNet for surface water mapping based on Sentinel-2 image, uses the [Earth Surface Water Dataset](https://zenodo.org/record/5205674#.YoMjyZPMK3I)

  2.3.21. [A-U-Net-for-Flood-Extent-Mapping](https://github.com/jorgemspereira/A-U-Net-for-Flood-Extent-Mapping) -> in keras

  2.3.22. [floatingobjects](https://github.com/ESA-PhiLab/floatingobjects) -> code for the paper: TOWARDS DETECTING FLOATING OBJECTS ON A GLOBAL SCALE WITHLEARNED SPATIAL FEATURES USING SENTINEL 2. Uses U-Net & pytorch

  2.3.23. [SpaceNet8](https://github.com/SpaceNetChallenge/SpaceNet8) -> baseline Unet solution to detect flooded roads and buildings

  2.3.24. [dlsim](https://github.com/nyokoya/dlsim) -> code for 2020 [paper](https://arxiv.org/abs/2006.05180): Breaking the Limits of Remote Sensing by Simulation and Deep Learning for Flood and Debris Flow Mapping

  2.3.25. [Water-HRNet](https://github.com/faye0078/Water-Extraction) -> HRNet trained on Sentinel 2

  2.3.26. [semantic segmentation model to identify newly developed or flooded land](https://github.com/Azure/pixel_level_land_classification) using NAIP imagery provided by the Chesapeake Conservancy, training on MS Azure

  2.3.27. [BandNet](https://github.com/IamShubhamGupto/BandNet) -> code for 2022 [paper](https://arxiv.org/abs/2212.08749): Analysis and application of multispectral data for water segmentation using machine learning. Uses Sentinel-2 data

  2.3.28. [mmflood](https://github.com/edornd/mmflood) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9882096): MMFlood: A Multimodal Dataset for Flood Delineation From Satellite Imagery (Sentinel 1 SAR)

  2.3.29. [Urban_flooding](https://github.com/omarseleem92/Urban_flooding) -> Towards transferable data-driven models to predict urban pluvial flood water depth in Berlin, Germany

  2.3.30. [Flood-Mapping-Using-Satellite-Images](https://github.com/KonstantinosF/Flood-Mapping-Using-Satellite-Images) -> masters thesis comparing Random Forest & Unet

### 2.4. Segmentation - Fire, smoke & burn areas

  2.4.1. [SatelliteVu-AWS-Disaster-Response-Hackathon](https://github.com/SatelliteVu/SatelliteVu-AWS-Disaster-Response-Hackathon) -> fire spread prediction using classical ML & deep learning `BEGINNER`

  2.4.2. [Wild Fire Detection](https://github.com/yueureka/WildFireDetection) using U-Net trained on Databricks & Keras, semantic segmentation

  2.4.3. [A Practical Method for High-Resolution Burned Area Monitoring Using Sentinel-2 and VIIRS](https://www.mdpi.com/2072-4292/13/9/1608) with [code](https://github.com/mnpinto/FireHR). Dataset created on Google Earth Engine, downloaded to local machine for model training using fastai. The BA-Net model used is much smaller than U-Net, resulting in lower memory requirements and a faster computation

  2.4.4. [AI Geospatial Wildfire Risk Prediction](https://towardsdatascience.com/ai-geospatial-wildfire-risk-prediction-8c6b1d415eb4) -> A predictive model using geospatial raster data to asses wildfire hazard potential over the contiguous United States using Unet

  2.4.5. [IndustrialSmokePlumeDetection](https://github.com/HSG-AIML/IndustrialSmokePlumeDetection) -> using Sentinel-2 & a modified ResNet-50

  2.4.6. [burned-area-detection](https://github.com/dymaxionlabs/burned-area-detection) -> uses Sentinel-2

  2.4.7. [rescue](https://github.com/dbdmg/rescue) -> code of the paper: Attention to fires: multi-channel deep-learning models forwildfire severity prediction

  2.4.8. [smoke_segmentation](https://github.com/jeffwen/smoke_segmentation) -> Segmenting smoke plumes and predicting density from GOES imagery

  2.4.9. [wildfire-detection](https://github.com/amanbasu/wildfire-detection) -> Using Vision Transformers for enhanced wildfire detection in satellite images

  2.4.10. [Burned_Area_Detection](https://github.com/prhuppertz/Burned_Area_Detection) -> Detecting Burned Areas with Sentinel-2 data

  2.4.11. [burned-area-baseline](https://github.com/lccol/burned-area-baseline) -> baseline unet model accompanying the Satellite Burned Area Dataset (Sentinel 1 & 2)

  2.4.12. [burned-area-seg](https://github.com/links-ads/burned-area-seg) -> Burned area segmentation from Sentinel-2 using multi-task learning

  2.4.13. [chabud2023](https://github.com/developmentseed/chabud2023) -> Change detection for Burned area Delineation (ChaBuD) ECML/PKDD 2023 challenge

  2.4.14. [Post Wildfire Burnt-up Detection using Siamese-UNet](https://github.com/kavyagupta/chabud) -> on Chadbud dataset

### 2.5.  Segmentation - Landslides

  2.5.1. [landslide-sar-unet](https://github.com/iprapas/landslide-sar-unet) -> code for 2022 [paper](https://arxiv.org/abs/2211.02869): Deep Learning for Rapid Landslide Detection using Synthetic Aperture Radar (SAR) Datacubes

  2.5.2. [landslide-mapping-with-cnn](https://github.com/nprksh/landslide-mapping-with-cnn) -> code for 2021 [paper](https://www.nature.com/articles/s41598-021-89015-8): A new strategy to map landslides with a generalized convolutional neural network

  2.5.3. [Relict_landslides_CNN_kmeans](https://github.com/SPAMLab/data_sharing/tree/main/Relict_landslides_CNN_kmeans) -> code for 2022 [paper](https://arxiv.org/abs/2208.02693): Relict landslide detection in rainforest areas using a combination of k-means clustering algorithm and Deep-Learning semantic segmentation models

  2.5.4. [Landslide-mapping-on-SAR-data-by-Attention-U-Net](https://github.com/lorenzonava96/Landslide-mapping-on-SAR-data-by-Attention-U-Net) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/6/1449): Rapid Mapping of landslide on SAR data by Attention U-net

  2.5.5. [SAR-landslide-detection-pretraining](https://github.com/VMBoehm/SAR-landslide-detection-pretraining) -> code for the 2022 [paper](https://arxiv.org/abs/2211.09927): SAR-based landslide classification pretraining leads to better segmentation

### 2.6. Segmentation - Glaciers

  2.6.1. [HED-UNet](https://github.com/khdlr/HED-UNet) -> a model for simultaneous semantic segmentation and edge detection, examples provided are glacier fronts and building footprints using the Inria Aerial Image Labeling dataset

  2.6.2. [glacier_mapping](https://github.com/krisrs1128/glacier_mapping) -> Mapping glaciers in the Hindu Kush Himalaya, Landsat 7 images, Shapefile labels of the glaciers, Unet with dropout

  2.6.3. [glacier-detect-ML](https://github.com/mikeskaug/glacier-detect-ML) -> a simple logistic regression model to identify a glacier in Landsat satellite imagery

  2.6.4. [GlacierSemanticSegmentation](https://github.com/n9Mtq4/GlacierSemanticSegmentation) -> uses unet

  2.6.5. [Antarctic-fracture-detection](https://github.com/chingyaolai/Antarctic-fracture-detection) -> uses UNet with the MODIS Mosaic of Antarctica to detect surface fractures ([paper](https://www.nature.com/articles/s41586-020-2627-8#code-availability))

### 2.7. Segmentation - Other environmental

  2.7.1. [Detection of Open Landfills](https://github.com/dymaxionlabs/basurales) -> uses Sentinel-2 to detect large changes in the Normalized Burn Ratio (NBR)

  2.7.2. [sea_ice_remote_sensing](https://github.com/sum1lim/sea_ice_remote_sensing) -> Sea Ice Concentration classification

  2.7.3. [Methane-detection-from-hyperspectral-imagery](https://github.com/satish1901/Methane-detection-from-hyperspectral-imagery) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9093600): Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery

  2.7.4. [EddyNet](https://github.com/redouanelg/EddyNet) -> A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies

  2.7.5. [schisto-vegetation](https://github.com/deleo-lab/schisto-vegetation) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/6/1345): Deep Learning Segmentation of Satellite Imagery Identifies Aquatic Vegetation Associated with Snail Intermediate Hosts of Schistosomiasis in Senegal, Africa

  2.7.6. [earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer) -> code for 2022 [paper](https://www.amazon.science/publications/earthformer-exploring-space-time-transformers-for-earth-system-forecasting): Earthformer: exploring space-time transformers for earth system forecasting

  2.7.7. [weather4cast-2022](https://github.com/iarai/weather4cast-2022) -> Unet-3D baseline model for Weather4cast Rain Movie Prediction competition

  2.7.8. [WeatherFusionNet](https://github.com/Datalab-FIT-CTU/weather4cast-2022) -> code for [paper](https://arxiv.org/abs/2211.16824): WeatherFusionNet: Predicting Precipitation from Satellite Data. weather4cast-2022 1st place solution

  2.7.9. [marinedebrisdetector](https://github.com/MarcCoru/marinedebrisdetector) -> code for paper: Large-scale Detection of Marine Debris in Coastal Areas with Sentinel-2

  2.7.10. [kaggle-identify-contrails-4th](https://github.com/selimsef/kaggle-identify-contrails-4th) -> 4th place Solution, Google Research - Identify Contrails to Reduce Global Warming

### 2.8. Segmentation - Roads
Extracting roads is challenging due to the occlusions caused by other objects and the complex traffic environment

  2.8.1. [Road detection using semantic segmentation and albumentations for data augmention](https://towardsdatascience.com/road-detection-using-segmentation-models-and-albumentations-libraries-on-keras-d5434eaf73a8) using the Massachusetts Roads Dataset, U-net & Keras. With [code](https://github.com/Diyago/ML-DL-scripts/tree/master/DEEP%20LEARNING/segmentation/Segmentation%20pipeline) `BEGINNER`

  2.8.2. [ML_EPFL_Project_2](https://github.com/LucasBrazCappelo/ML_EPFL_Project_2) -> U-Net in Pytorch to perform semantic segmentation of roads on satellite images `BEGINNER`

  2.8.3. [Semantic Segmentation of roads](https://vihan-tyagi.medium.com/semantic-segmentation-of-satellite-images-based-on-deep-learning-algorithms-ea5ec408ac53) using  U-net Keras, OSM data, project summary article by student, no code

  2.8.4. [Winning Solutions from SpaceNet Road Detection and Routing Challenge](https://github.com/SpaceNetChallenge/RoadDetector)

  2.8.5. [RoadVecNet](https://github.com/gismodelling/RoadVecNet) -> Road-Network-Segmentation-and-Vectorization in keras with dataset and [paper](https://www.tandfonline.com/doi/abs/10.1080/15481603.2021.1972713?journalCode=tgrs20&)

  2.8.6. [Detecting road and road types jupyter notebook](https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb)

  2.8.7. [awesome-deep-map](https://github.com/antran89/awesome-deep-map) -> A curated list of resources dedicated to deep learning / computer vision algorithms for mapping. The mapping problems include road network inference, building footprint extraction, etc.

  2.8.8. [RoadTracer: Automatic Extraction of Road Networks from Aerial Images](https://github.com/mitroadmaps/roadtracer) -> uses an iterative search process guided by a CNN-based decision function to derive the road network graph directly from the output of the CNN

  2.8.9. [road_detection_mtl](https://github.com/ntelo007/road_detection_mtl) -> Road Detection using a multi-task Learning technique to improve the performance of the road detection task by incorporating prior knowledge constraints, uses the SpaceNet Roads Dataset

  2.8.10. [road_connectivity](https://github.com/anilbatra2185/road_connectivity) -> Improved Road Connectivity by Joint Learning of Orientation and Segmentation (CVPR2019)

  2.8.11. [Road-Network-Extraction using classical Image processing](https://github.com/abhaykes1/Road-Network-Extraction) -> blur & canny edge detection

  2.8.12. [SPIN_RoadMapper](https://github.com/wgcban/SPIN_RoadMapper) -> Extracting Roads from Aerial Images via Spatial and Interaction Space Graph Reasoning for Autonomous Driving

  2.8.13. [road_extraction_remote_sensing](https://github.com/jiankang1991/road_extraction_remote_sensing) -> pytorch implementation, CVPR2018 DeepGlobe Road Extraction Challenge submission. See also [DeepGlobe-Road-Extraction-Challenge](https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge)

  2.8.14. [RoadDetections dataset by Microsoft](https://github.com/microsoft/RoadDetections)

  2.8.15. [CoANet](https://github.com/mj129/CoANet) -> Connectivity Attention Network for Road Extraction From Satellite Imagery. The CoA module incorporates graphical information to ensure the connectivity of roads are better preserved. With [paper](https://ieeexplore.ieee.org/document/9563125)

  2.8.16. [Satellite Imagery Road Segmentation](https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812) -> intro articule on Medium using the kaggle [Massachusetts Roads Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset)

  2.8.17. [Label-Pixels](https://github.com/venkanna37/Label-Pixels) -> for semantic segmentation of roads and other features

  2.8.18. [Satellite-image-road-extraction](https://github.com/amanhari-projects/Satellite-image-road-extraction) -> code for 2018 paper: Road Extraction by Deep Residual U-Net

  2.8.19. [road_building_extraction](https://github.com/jeffwen/road_building_extraction) -> Pytorch implementation of U-Net architecture for road and building extraction

  2.8.20. [RCFSNet](https://github.com/CVer-Yang/RCFSNet) -> code for 2022 paper: Road Extraction From Satellite Imagery by Road Context and Full-Stage Feature

  2.8.21. [SGCN](https://github.com/tist0bsc/SGCN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9614130): Split Depth-Wise Separable Graph-Convolution Network for Road Extraction in Complex Environments From High-Resolution Remote-Sensing Images

  2.8.22. [ASPN](https://github.com/pshams55/ASPN) -> code for 2020 [paper](https://arxiv.org/abs/2008.04021): Road Segmentation for Remote Sensing Images using Adversarial Spatial Pyramid Networks

  2.8.23. [FCNs-for-road-extraction-keras](https://github.com/zetrun-liu/FCNs-for-road-extraction-keras) -> Road extraction of high-resolution remote sensing images based on various semantic segmentation networks

  2.8.24. [cresi](https://github.com/avanetten/cresi) -> Road network extraction from satellite imagery, with speed and travel time estimates

  2.8.25. [road-extraction-d-linknet](https://github.com/NekoApocalypse/road-extraction-d-linknet) -> code for 2018 [paper](https://ieeexplore.ieee.org/document/8575492): D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction

  2.8.26. [Sat2Graph](https://github.com/songtaohe/Sat2Graph) -> code for 2020 paper: Road Graph Extraction through Graph-Tensor Encoding

  2.8.27. [Image-Segmentation)](https://github.com/mschulz/Image-Segmentation) -> using Massachusetts Road dataset and fast.ai

  2.8.28. [RoadTracer-M](https://github.com/astro-ck/RoadTracer-M) -> code for 2019 [paper](https://ieeexplore.ieee.org/abstract/document/8898565): Road Network Extraction from Satellite Images Using CNN Based Segmentation and Tracing

  2.8.29. [ScRoadExtractor](https://github.com/weiyao1996/ScRoadExtractor) -> code for 2020 [paper](https://arxiv.org/abs/2010.13106): Scribble-based Weakly Supervised Deep Learning for Road Surface Extraction from Remote Sensing Images

  2.8.30. [RoadDA](https://github.com/LANMNG/RoadDA) -> code for 2021 [paper](https://arxiv.org/abs/2108.12611): Stagewise Unsupervised Domain Adaptation with Adversarial Self-Training for Road Segmentation of Remote Sensing Images

  2.8.31. [DeepSegmentor](https://github.com/yhlleo/DeepSegmentor) -> A Pytorch implementation of DeepCrack and RoadNet projects

  2.8.32. [Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction](https://github.com/liaochengcsu/Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction) -> code for 2021 [paper](https://www.mdpi.com/2220-9964/11/1/9): Cascaded Residual Attention Enhanced Road Extraction from Remote Sensing Images

  2.8.33. [nia-road-baseline](https://github.com/SIAnalytics/nia-road-baseline) -> code for 2020 [paper](https://arxiv.org/abs/1908.08223): NL-LinkNet: Toward Lighter but More Accurate Road Extraction with Non-Local Operations

  2.8.34. [IRSR-net](https://github.com/yangzhen1252/IRSR-net) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9785827): Lightweight Remote Sensing Road Detection Network

  2.8.35. [hironex](https://github.com/johannesuhl/hironex) -> A python tool for automatic, fully unsupervised extraction of historical road networks from historical maps

  2.8.36. [Road_detection_model](https://github.com/JonasImazon/Road_detection_model) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/15/3625): Mapping Roads in the Brazilian Amazon with Artificial Intelligence and Sentinel-2

  2.8.37. [DTnet](https://github.com/huzican695/DTnet) -> code for 2022 [paper](https://arxiv.org/abs/2208.08116): Road detection via a dual-task network based on cross-layer graph fusion modules

  2.8.38. [Automatic-Road-Extraction-from-Historical-Maps-using-Deep-Learning-Techniques](https://github.com/UrbanOccupationsOETR/Automatic-Road-Extraction-from-Historical-Maps-using-Deep-Learning-Techniques) -> code for the paper: Automatic Road Extraction from Historical Maps using Deep Learning Techniques: A Regional Case Study of Turkey in a German World War II map

  2.8.39. [Istanbul_Dataset](https://github.com/TolgaBkm/Istanbul_Dataset) -> segmentation on the Istanbul, Inria and Massachusetts datasets

  2.8.40. [Road-Segmentation](https://github.com/ralph-elhaddad/Road-Segmentation) -> Road segmentation on Satellite Images using CNN (U-Nets and FCN8) and Logistic Regression

  2.8.41. [D-LinkNet](https://github.com/ShenweiXie/D-LinkNet) -> 1st place solution in DeepGlobe Road Extraction Challenge

  2.8.42. [PaRK-Detect](https://github.com/ShenweiXie/PaRK-Detect) -> code for 2023 paper: PaRK-Detect: Towards Efficient Multi-Task Satellite Imagery Road Extraction via Patch-Wise Keypoints Detection

  2.8.43. [tile2net](https://github.com/VIDA-NYU/tile2net) -> code for 2023 paper: Mapping the walk: A scalable computer vision approach for generating sidewalk network datasets from aerial imagery

### 2.9. Segmentation - Buildings & rooftops

  2.9.1. [Road and Building Semantic Segmentation in Satellite Imagery](https://github.com/Paulymorphous/Road-Segmentation) uses U-Net on the Massachusetts Roads Dataset & keras `BEGINNER`

  2.9.2. [find-unauthorized-constructions-using-aerial-photography](https://medium.com/towards-artificial-intelligence/find-unauthorized-constructions-using-aerial-photography-and-deep-learning-with-code-part-2-b56ca80c8c99) -> semantic segmentation using U-Net with custom_f1 metric & Keras. The creation of the dataset is described in [this article](https://pub.towardsai.net/find-unauthorized-constructions-using-aerial-photography-and-deep-learning-with-code-part-1-6d3ca7ff6fa0) `BEGINNER`

  2.9.3. [Semantic Segmentation on Aerial Images using fastai](https://medium.com/swlh/semantic-segmentation-on-aerial-images-using-fastai-a2696e4db127) uses U-Net on the Inria Aerial Image Labeling Dataset of urban settlements in Europe and the United States, and is labelled as a building and not building classes (no repo) `BEGINNER`

  2.9.4. [Building footprint detection with fastai on the challenging SpaceNet7 dataset](https://deeplearning.berlin/satellite%20imagery/computer%20vision/fastai/2021/02/17/Building-Detection-SpaceNet7.html) uses U-Net & fastai `BEGINNER`

  2.9.5. [Pix2Pix-for-Semantic-Segmentation-of-Satellite-Images](https://github.com/A2Amir/Pix2Pix-for-Semantic-Segmentation-of-Satellite-Images) -> using Pix2Pix GAN network to segment the building footprint from Satellite Images, uses tensorflow

  2.9.6. [SpaceNetUnet](https://github.com/boggis30/SpaceNetUnet) -> Baseline model is U-net like, applied to SpaceNet Vegas data, using Keras

  2.9.7. [automated-building-detection](https://github.com/rodekruis/automated-building-detection) -> Input: very-high-resolution (<= 0.5 m/pixel) RGB satellite images. Output: buildings in vector format (geojson), to be used in digital map products. Built on top of robosat and robosat.pink.

  2.9.8. [project_sunroof_india](https://github.com/AKASH2907/project_sunroof_india) -> Analyzed Google Satellite images to generate a report on individual house rooftop's solar power potential, uses a range of classical computer vision techniques (e.g Canny Edge Detection) to segment the roofs

  2.9.9. [JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction)

  2.9.10. [Mapping Africa’s Buildings with Satellite Imagery: Google AI blog post](https://ai.googleblog.com/2021/07/mapping-africas-buildings-with.html). See the [open-buildings](https://sites.research.google/open-buildings/) dataset

  2.9.11. [nz_convnet](https://github.com/weiji14/nz_convnet) -> A U-net based ConvNet for New Zealand imagery to classify building outlines

  2.9.12. [polycnn](https://github.com/Lydorn/polycnn) -> End-to-End Learning of Polygons for Remote Sensing Image Classification

  2.9.13. [spacenet_building_detection](https://github.com/motokimura/spacenet_building_detection) solution by [motokimura](https://github.com/motokimura) using Unet

<!-- markdown-link-check-disable -->
  2.9.14. [How to extract building footprints from satellite images using deep learning](https://azure.microsoft.com/en-us/blog/how-to-extract-building-footprints-from-satellite-images-using-deep-learning/)
<!-- markdown-link-check-enable -->

  2.9.15. [Vec2Instance](https://github.com/lakmalnd/Vec2Instance) -> applied to the SpaceNet challenge AOI 2 (Vegas) building footprint dataset, tensorflow v1.12

  2.9.16. [EarthquakeDamageDetection](https://github.com/JaneKravchenko/EarthquakeDamageDetection) -> Buildings segmentation from satellite imagery and damage classification for each build, using Keras

  2.9.17. [Semantic-segmentation repo by fuweifu-vtoo](https://github.com/fuweifu-vtoo/Semantic-segmentation) -> uses pytorch and the [Massachusetts Buildings & Roads Datasets](https://www.cs.toronto.edu/~vmnih/data/)

  2.9.18. [Extracting buildings and roads from AWS Open Data using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/extracting-buildings-and-roads-from-aws-open-data-using-amazon-sagemaker/) -> uses merged RGB (SpaceNet) and LiDAR (USGS 3DEP) datasets with Unet to reproduce the winning algorithm from SpaceNet challenge 4 by XD_XD. With [repo](https://github.com/aws-samples/aws-open-data-satellite-lidar-tutorial)

  2.9.19. [TF-SegNet](https://github.com/mathildor/TF-SegNet) -> AirNet is a segmentation network based on SegNet, but with some modifications

  2.9.20. [rgb-footprint-extract](https://github.com/aatifjiwani/rgb-footprint-extract) -> a Semantic Segmentation Network for Urban-Scale Building Footprint Extraction Using RGB Satellite Imagery, DeepLavV3+ module with a Dilated ResNet C42 backbone

  2.9.21. [SpaceNetExploration](https://github.com/yangsiyu007/SpaceNetExploration) -> A sample project demonstrating how to extract building footprints from satellite images using a semantic segmentation model. Data from the SpaceNet Challenge

  2.9.22. [Rooftop-Instance-Segmentation](https://github.com/MasterSkepticista/Rooftop-Instance-Segmentation) -> VGG-16, Instance Segmentation, uses the Airs dataset

  2.9.23. [solar-farms-mapping](https://github.com/microsoft/solar-farms-mapping) -> An Artificial Intelligence Dataset for Solar Energy Locations in India

  2.9.24. [poultry-cafos](https://github.com/microsoft/poultry-cafos) -> This repo contains code for detecting poultry barns from high-resolution aerial imagery and an accompanying dataset of predicted barns over the United States

  2.9.25. [ssai-cnn](https://github.com/mitmul/ssai-cnn) -> This is an implementation of Volodymyr Mnih's dissertation methods on his Massachusetts road & building dataset

  2.9.26. [Remote-sensing-building-extraction-to-3D-model-using-Paddle-and-Grasshopper](https://github.com/Youssef-Harby/Remote-sensing-building-extraction-to-3D-model-using-Paddle-and-Grasshopper)

  2.9.27. [segmentation-enhanced-resunet](https://github.com/tranleanh/segmentation-enhanced-resunet) -> Urban building extraction in Daejeon region using Modified Residual U-Net (Modified ResUnet) and applying post-processing

  2.9.28. [Mask RCNN for Spacenet Off Nadir Building Detection](https://github.com/ashnair1/Mask-RCNN-for-Off-Nadir-Building-Detection)

  2.9.29. [GRSL_BFE_MA](https://github.com/jiankang1991/GRSL_BFE_MA) -> Deep Learning-based Building Footprint Extraction with Missing Annotations using a novel loss function

  2.9.30. [FER-CNN](https://github.com/runnergirl13/FER-CNN) -> Detection, Classification and Boundary Regularization of Buildings in Satellite Imagery Using Faster Edge Region Convolutional Neural Networks, with [paper](https://www.mdpi.com/2072-4292/12/14/2240/htm)

  2.9.31. [UNET-Image-Segmentation-Satellite-Picture](https://github.com/rwie1and/UNET-Image-Segmentation-Satellite-Pictures) -> Unet to predict roof tops on Crowed AI Mapping dataset, uses keras

  2.9.32. [Vector-Map-Generation-from-Aerial-Imagery-using-Deep-Learning-GeoSpatial-UNET](https://github.com/ManishSahu53/Vector-Map-Generation-from-Aerial-Imagery-using-Deep-Learning-GeoSpatial-UNET) -> applied to geo-referenced images which are very large size > 10k x 10k pixels

  2.9.33. [building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation) -> pip installable library to train building footprint segmentation on satellite and aerial imagery, applied to Massachusetts Buildings Dataset and Inria Aerial Image Labeling Dataset

  2.9.34. [SemSegBuildings](https://github.com/SharpestProjects/SemSegBuildings) -> Project using fast.ai framework for semantic segmentation on Inria building segmentation dataset

  2.9.35. [FCNN-example](https://github.com/emredog/FCNN-example) -> overfit to a given single image to detect houses

  2.9.36. [SAT2LOD2](https://github.com/gdaosu/lod2buildingmodel) -> an open-source, python-based GUI-enabled software that takes the satellite images as inputs and returns LoD2 building models as outputs, with [paper](https://arxiv.org/abs/2204.04139)

  2.9.37. [SatFootprint](https://github.com/PriyanK7n/SatFootprint) -> building segmentation on the Spacenet 7 dataset

  2.9.38. [Building-Detection](https://github.com/EL-BID/Building-Detection) -> code for running a Raster Vision experiment to train a model to detect buildings from satellite imagery in three cities in Latin America

  2.9.39. [Multi-building-tracker](https://github.com/sebasmos/Multi-building-tracker) -> code for paper: Multi-target building tracker for satellite images using deep learning

  2.9.40. [Boundary Enhancement Semantic Segmentation for Building Extraction](https://github.com/hin1115/BEmodule-Satellite-Building-Segmentation)

  2.9.41. [UNet_keras_for_RSimage](https://github.com/loveswine/UNet_keras_for_RSimage) -> keras code for binary semantic segmentation

  2.9.42. [Spacenet-Building-Detection](https://github.com/IdanC1s2/Spacenet-Building-Detection) -> uses keras

  2.9.43. [LGPNet-BCD](https://github.com/TongfeiLiu/LGPNet-BCD) -> code for 2021 paper: Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy

  2.9.44. [MTL_homoscedastic_SRB](https://github.com/burakekim/MTL_homoscedastic_SRB) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9554766): A Multi-Task Deep Learning Framework for Building Footprint Segmentation

  2.9.45. [UNet_CNN](https://github.com/Inamdarpushkar/UNet_CNN) -> UNet model to segment building coverage in Boston using Remote sensing data, uses keras

  2.9.46. [FDANet](https://github.com/daifeng2016/FDANet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9481881): Full-Level Domain Adaptation for Building Extraction in Very-High-Resolution Optical Remote-Sensing Images

  2.9.47. [CBRNet](https://github.com/HaonanGuo/CBRNet) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002975): A Coarse-to-fine Boundary Refinement Network for Building Extraction from Remote Sensing Imagery

  2.9.48. [ASLNet](https://github.com/ggsDing/ASLNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9653801): Adversarial Shape Learning for Building Extraction in VHR Remote Sensing Images

  2.9.49. [BRRNet](https://github.com/wangyi111/Building-Extraction) -> implementation of Modified U-Net from 2020 [paper](https://www.mdpi.com/2072-4292/12/6/1050): BRRNet: A Fully Convolutional Neural Network for Automatic Building Extraction From High-Resolution Remote Sensing Images

  2.9.50. [Multi-Scale-Filtering-Building-Index](https://github.com/ThomasWangWeiHong/Multi-Scale-Filtering-Building-Index) -> Python implementation of building extraction index proposed in 2019 [paper](https://www.mdpi.com/2072-4292/11/5/482): A Multi - Scale Filtering Building Index for Building Extraction in Very High - Resolution Satellite Imagery

  2.9.51. [Models for Remote Sensing](https://github.com/bohaohuang/mrs) -> long list of unets etc applied to building detection

  2.9.52. [boundary_loss_for_remote_sensing](https://github.com/yiskw713/boundary_loss_for_remote_sensing) -> code for 2019 paper: Boundary Loss for Remote Sensing Imagery
  Semantic Segmentation

  2.9.53. [Open Cities AI Challenge](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/) -> Segmenting Buildings for Disaster Resilience. Winning solutions [on Github](https://github.com/drivendataorg/open-cities-ai-challenge/)

  2.9.54. [MAPNet](https://github.com/lehaifeng/MAPNet) -> code for 2020 [paper](https://arxiv.org/abs/1910.12060): Multi Attending Path Neural Network for Building Footprint Extraction from Remote Sensed Imagery

  2.9.55. [dual-hrnet](https://github.com/SIAnalytics/dual-hrnet) -> localizing buildings and classifying their damage level

  2.9.56. [ESFNet](https://github.com/mrluin/ESFNet-Pytorch) -> code for 2019 [paper](https://arxiv.org/abs/1903.12337): Efficient Network for Building Extraction from High-Resolution Aerial Images

  2.9.57. [rooftop-detection-python](https://github.com/sayonpalit/rooftop-detection-python) -> Detect Rooftops from low resolution satellite images and calculate area for cultivation and solar panel installment using classical computer vision techniques

  2.9.58. [keras_segmentation_models](https://github.com/sajmonogy/keras_segmentation_models) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/12/2745): Using Open Vector-Based Spatial Data to Create Semantic Datasets for Building Segmentation for Raster Data

  2.9.59. [CVCMFFNet](https://github.com/Jiankun-chen/CVCMFFNet-master) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9397870): Complex-Valued Convolutional and Multifeature Fusion Network for Building Semantic Segmentation of InSAR Images

  2.9.60. [STEB-UNet](https://github.com/BrightGuo048/STEB-UNet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/11/2611): A Swin Transformer-Based Encoding Booster Integrated in U-Shaped Network for Building Extraction

  2.9.61. [dfc2020_baseline](https://github.com/lukasliebel/dfc2020_baseline) -> Baseline solution for the IEEE GRSS Data Fusion Contest 2020. Predict land cover labels from Sentinel-1 and Sentinel-2 imagery. Code for 2020 [paper](https://arxiv.org/abs/2002.08254): Weakly Supervised Semantic Segmentation of Satellite Images for Land Cover Mapping

  2.9.62. [Fusing multiple segmentation models based on different datasets into a single edge-deployable model](https://github.com/markusmeingast/Satellite-Classifier) -> roof, car & road segmentation

  2.9.63. [ground-truth-gan-segmentation](https://github.com/zakariamejdoul/ground-truth-gan-segmentation) -> use Pix2Pix to segment the footprint of a building. The dataset used is AIRS

  2.9.64. [UNICEF-Giga_Sudan](https://github.com/Kamal-Eldin/UNICEF-Giga_Sudan) -> Detecting school lots from satellite imagery in Southern Sudan using a UNET segmentation model

  2.9.65. [building_footprint_extraction](https://github.com/shubhamgoel27/building_footprint_extraction) -> The project retrieves satellite imagery from Google and performs building footprint extraction using a U-Net.

  2.9.66. [projectRegularization](https://github.com/zorzi-s/projectRegularization) -> code for 2019 [paper](https://arxiv.org/abs/2007.11840): Regularization of building boundaries in satellite images using adversarial and regularized losses

  2.9.67. [PolyWorldPretrainedNetwork](https://github.com/zorzi-s/PolyWorldPretrainedNetwork) -> code for 2021 [paper](https://arxiv.org/abs/2111.15491): Polygonal Building Extraction with Graph Neural Networks in Satellite Images

  2.9.68. [dl_image_segmentation](https://github.com/harry-gibson/dl_image_segmentation) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/13/3072): Uncertainty-Aware Interpretable Deep Learning for Slum Mapping and Monitoring. Uses SHAP

  2.9.69. [UBC-dataset](https://github.com/AICyberTeam/UBC-dataset) -> a dataset for building detection and classification from very high-resolution satellite imagery with the focus on object-level interpretation of individual buildings

  2.9.70. [GeoSeg](https://github.com/WangLibo1995/GeoSeg) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S0924271622001654): UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery

  2.9.71. [BESNet](https://github.com/FlyC235/BESNet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/7/1638): BES-Net: Boundary Enhancing Semantic Context Network for High-Resolution Image Semantic Segmentation. Applied to Vaihingen and Potsdam datasets

  2.9.72. [CVNet](https://github.com/xzq-njust/CVNet) -> code for 2022 paper: CVNet: Contour Vibration Network for Building Extraction

  2.9.73. [CFENet](https://github.com/djzgroup/CFENet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/9/2276): A Context Feature Enhancement Network for Building Extraction from High-Resolution Remote Sensing Imagery

  2.9.74. [HiSup](https://github.com/SarahwXU/HiSup) -> code for 2022 [paper](https://arxiv.org/abs/2208.00609): Accurate Polygonal Mapping of Buildings in Satellite Imagery

  2.9.75. [BuildingExtraction](https://github.com/KyanChen/BuildingExtraction) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/21/4441): Building Extraction from Remote Sensing Images with Sparse Token Transformers

  2.9.76. [coseg_building](https://github.com/lqycrystal/coseg_building) -> code for the 2022 [paper](https://www.sciencedirect.com/science/article/pii/S1569843222000267): CrossGeoNet: A Framework for Building Footprint Generation of Label-Scarce Geographical Regions

  2.9.77. [AFM_building](https://github.com/lqycrystal/AFM_building) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9538384): Building Footprint Generation Through Convolutional Neural Networks With Attraction Field Representation

  2.9.78. [ramp-code](https://github.com/devglobalpartners/ramp-code) -> code for the RAMP (Replicable AI for MicroPlanning) project, which enables building detection in low and middle income countries

  2.9.79. [Building-instance-segmentation](https://github.com/yuanqinglie/Building-instance-segmentation-combining-anchor-free-detectors-and-multi-modal-feature-fusion) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/19/4920): Multi-Modal Feature Fusion Network with Adaptive Center Point Detector for Building Instance Extraction

  2.9.80. [CGSANet](https://github.com/MrChen18/CGSANet) -> code for the 2021 [paper](https://ieeexplore.ieee.org/document/9664368): CGSANet: A Contour-Guided and Local Structure-Aware Encoder–Decoder Network for Accurate Building Extraction From Very High-Resolution Remote Sensing Imagery

  2.9.81. [building-footprints-update](https://github.com/wangzehui20/building-footprints-update) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/22/5851): Learning Color Distributions from Bitemporal Remote Sensing Images to Update Existing Building Footprints

  2.9.82. [Istanbul_Dataset](https://github.com/TolgaBkm/Istanbul_Dataset) -> this repo contains weights of Unet++ model with SE-ResNeXt101 encoder trained with Istanbul, Inria and Massachusetts datasets seperately. Accompanies the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417422007011?via%3Dihub): Comparative analysis of deep learning based building extraction methods with the new VHR Istanbul dataset

  2.9.83. [RAMP](https://rampml.global/) -> model and buildings dataset to support a wide variety of humanitarian use cases

  2.9.84. [Thesis_Semantic_Image_Segmentation_on_Satellite_Imagery_using_UNets](https://github.com/rinkwitz/Thesis_Semantic_Image_Segmentation_on_Satellite_Imagery_using_UNets) -> This master thesis aims to perform semantic segmentation of buildings on satellite images from the SpaceNet challenge 1 dataset using the U-Net architecture

### 2.10. Segmentation - Solar panels

  2.10.1. [DeepSolar](https://github.com/wangzhecheng/DeepSolar) -> A Machine Learning Framework to Efficiently Construct a Solar Deployment Database in the United States. [Dataset on kaggle](https://www.kaggle.com/tunguz/deep-solar-dataset), actually used a CNN for classification and segmentation is obtained by applying a threshold to the activation map. Original code is tf1 but [tf2/kers](https://github.com/aidan-fitz/deepsolar-v2) and a [pytorch implementation](https://github.com/wangzhecheng/deepsolar_pytorch) are available. Also checkout [Visualizations and in-depth analysis .. of the factors that can explain the adoption of solar energy in ..  Virginia](https://github.com/bessammehenni/DeepSolar_adoption_Virginia) and [DeepSolar tracker: towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping](https://github.com/gabrielkasmi/dsfrance)

  2.10.2. [hyperion_solar_net](https://github.com/fvergaracontesse/hyperion_solar_net) -> trained classificaton & segmentation models on RGB imagery from Google Maps. Provides app for viewing predictions, and has [arxiv paper](https://arxiv.org/abs/2201.02107)

  2.10.3. [3D-PV-Locator](https://github.com/kdmayer/3D-PV-Locator) -> Large-scale detection of rooftop-mounted photovoltaic systems in 3D

  2.10.4. [PV_Pipeline](https://github.com/kdmayer/PV_Pipeline) -> PyTorch models and pipeline developed for "DeepSolar for Germany"

  2.10.5. [solar-panels-detection](https://github.com/dbaofd/solar-panels-detection) -> using SegNet, Fast SCNN & ResNet

  2.10.6. [predict_pv_yield](https://github.com/openclimatefix/predict_pv_yield) -> Using optical flow & machine learning to predict PV yield

  2.10.7. [Large-scale-solar-plant-monitoring](https://github.com/osmarluiz/Large-scale-solar-plant-monitoring) -> code for the paper "Remote Sensing for Monitoring of Photovoltaic Power Plants in Brazil Using Deep Semantic Segmentation"

  2.10.8. [Panel-Segmentation](https://github.com/NREL/Panel-Segmentation) -> Determine the presence of a solar array in the satellite image (boolean True/False), using a VGG16 classification model

  2.10.9. [Roofpedia](https://github.com/ualsg/Roofpedia) -> an open registry of green roofs and solar roofs across the globe identified by Roofpedia through deep learning

  2.10.10. [Predicting the Solar Potential of Rooftops using Image Segmentation and Structured Data](https://medium.com/nam-r/predicting-the-solar-potential-of-rooftops-using-image-segmentation-and-structured-data-61198c39d57c) Medium article, using 20cm imagery & Unet

  2.10.11. [solar-pv-global-inventory](https://github.com/Lkruitwagen/solar-pv-global-inventory) -> code from the Nature paper of Kruitwagen et al, used to produce a global inventory of utility-scale solar photvoltaic generating stations

  2.10.12. [remote-sensing-solar-pv](https://github.com/Lkruitwagen/remote-sensing-solar-pv) -> A repository for sharing progress on the automated detection of solar PV arrays in sentinel-2 remote sensing imagery

  2.10.13. [solar-panel-segmentation)](https://github.com/gabrieltseng/solar-panel-segmentation) -> Finding solar panels using USGS satellite imagery

  2.10.14. [solar_seg](https://github.com/tcapelle/solar_seg) -> Solar segmentation of PV modules (sub elements of panels) using drone images and fast.ai

  2.10.15. [solar_plant_detection](https://github.com/Amirmoradi94/solar_plant_detection) -> boundary extraction of Photovoltaic (PV) plants using Mask RCNN and Amir dataset

  2.10.16. [SolarDetection](https://github.com/A-Stangeland/SolarDetection) -> unet on satellite image from the USA and France

  2.10.17. [adopptrs](https://github.com/francois-rozet/adopptrs) -> Automatic Detection Of Photovoltaic Panels Through Remote Sensing using unet & pytorch

  2.10.18. [solar-panel-locator](https://github.com/TorrBorr/solar-panel-locator) -> the number of solar panel pixels was only ~0.2% of the total pixels in the dataset, so solar panel data was upsampled to account for the class imbalance

  2.10.19. [projects-solar-panel-detection](https://github.com/top-on/projects-solar-panel-detection) -> List of project to detect solar panels from aerial/satellite images

  2.10.20. [Satellite_ComputerVision](https://github.com/mjevans26/Satellite_ComputerVision) -> UNET to detect solar arrays from Sentinel-2 data, using Google Earth Engine and Tensorflow. Also covers parking lot detection

  2.10.21. [photovoltaic-detection](https://github.com/riccardocadei/photovoltaic-detection) -> Detecting available rooftop area from satellite images to install photovoltaic panels

  2.10.22. [Solar_UNet](https://github.com/mjevans26/Solar_UNet) -> U-Net models delineating solar arrays in Sentinel-2 imagery

### 2.11. Segmentation - Other manmade

  2.11.1. [Aarsh2001/ML_Challenge_NRSC](https://github.com/Aarsh2001/ML_Challenge_NRSC) -> Electrical Substation detection

  2.11.2. [electrical_substation_detection](https://github.com/thisishardik/electrical_substation_detection) -> using UNet, Albumentations for image augmentation, and OpenCV for computer vision tasks

  2.11.3. [PLGAN-for-Power-Line-Segmentation](https://github.com/R3ab/PLGAN-for-Power-Line-Segmentation) -> code for 2022 [paper](https://arxiv.org/abs/2204.07243): PLGAN: Generative Adversarial Networks for Power-Line Segmentation in Aerial Images

  2.11.4. [MCAN-OilSpillDetection](https://github.com/liyongqingupc/MCAN-OilSpillDetection) -> Oil Spill Detection with A Multiscale Conditional Adversarial Network under Small Data Training, with [paper](https://www.mdpi.com/2072-4292/13/12/2378). A multiscale conditional adversarial network (MCAN) trained with four oil spill observation images accurately detects oil spills in new images.

  2.11.5. [plastics](https://github.com/earthrise-media/plastics) -> Detecting and Monitoring Plastic Waste Aggregations in Sentinel-2 Imagery for [globalplasticwatch.org](https://globalplasticwatch.org/)

  2.11.6. [mining-detector](https://github.com/earthrise-media/mining-detector) -> detection of artisanal gold mines in Sentinel-2 satellite imagery for [Amazon Mining Watch](https://amazonminingwatch.org/). Also covers clandestine airstrips

  2.11.7. [EG-UNet](https://github.com/tist0bsc/EG-UNet) code for 2023 paper: Deep Feature Enhancement Method for Land Cover With Irregular and Sparse Spatial Distribution Features: A Case Study on Open-Pit Mining

### 2.12. Panoptic segmentation

  2.12.1. [Things and stuff or how remote sensing could benefit from panoptic segmentation](https://softwaremill.com/things-and-stuff-or-how-remote-sensing-could-benefit-from-panoptic-segmentation/)

  2.12.2. [Panoptic Segmentation Meets Remote Sensing (paper)](https://www.mdpi.com/2072-4292/14/4/965)

  2.12.3. [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)

  2.12.4. [Panoptic-Generator](https://github.com/abilius-app/Panoptic-Generator) -> This module converts GIS data into panoptic segmentation tiles

  2.12.5. [BSB-Aerial-Dataset](https://github.com/osmarluiz/BSB-Aerial-Dataset) -> an example on how to use Detectron2's Panoptic-FPN in the BSB Aerial Dataset

  2.12.6. [utae-paps](https://github.com/VSainteuf/utae-paps) -> PyTorch implementation of U-TAE and PaPs for satellite image time series panoptic segmentation

### 2.13. Segmentation - Miscellaneous

  2.13.1. [awesome-satellite-images-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation#satellite-images-segmentation)

  2.13.2. [Satellite Image Segmentation: a Workflow with U-Net](https://medium.com/vooban-ai/satellite-image-segmentation-a-workflow-with-u-net-7ff992b2a56e) is a decent intro article `BEGINNER`

  2.13.3. [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) -> Semantic Segmentation Toolbox with support for many remote sensing datasets including LoveDA
  , Potsdam, Vaihingen & iSAID

  2.13.4. [segmentation_gym](https://github.com/Doodleverse/segmentation_gym) -> A neural gym for training deep learning models to carry out geoscientific image segmentation

  2.13.5. [How to create a DataBlock for Multispectral Satellite Image Semantic Segmentation using Fastai](https://towardsdatascience.com/how-to-create-a-datablock-for-multispectral-satellite-image-segmentation-with-the-fastai-v2-bc5e82f4eb5)

  2.13.6. [Using a U-Net for image segmentation, blending predicted patches smoothly is a must to please the human eye](https://github.com/Vooban/Smoothly-Blend-Image-Patches) -> python code to blend predicted patches smoothly. See [Satellite-Image-Segmentation-with-Smooth-Blending](https://github.com/MaitrySinha21/Satellite-Image-Segmentation-with-Smooth-Blending)

  2.13.7. [DCA](https://github.com/Luffy03/DCA) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9745130): Deep Covariance Alignment for Domain Adaptive Remote Sensing Image Segmentation

  2.13.8. [SCAttNet](https://github.com/lehaifeng/SCAttNet) -> Semantic Segmentation Network with Spatial and Channel Attention Mechanism

  2.13.9. [unetseg](https://github.com/dymaxionlabs/unetseg) -> A set of classes and CLI tools for training a semantic segmentation model based on the U-Net architecture, using Tensorflow and Keras. This implementation is tuned specifically for satellite imagery and other geospatial raster data

  2.13.10. [Semantic Segmentation of Satellite Imagery using U-Net & fast.ai](https://medium.com/dataseries/image-semantic-segmentation-of-satellite-imagery-using-u-net-e99ae13cf464) -> with [repo](https://github.com/raoofnaushad/Image-Semantic-Segmentation-of-Satellite-Imagery-using-U-Net.)

  2.13.11. [clusternet_segmentation](https://github.com/zhygallo/clusternet_segmentation) -> Unsupervised Segmentation by applying K-Means clustering to the features generated by Neural Network

  2.13.12. [Collection of different Unet Variant](https://github.com/ashishpatel26/satellite-Image-Semantic-Segmentation-Unet-Tensorflow-keras) -> demonstrates VggUnet, ResUnet, DenseUnet, Unet. AttUnet, MobileNetUnet, NestedUNet, R2AttUNet, R2UNet, SEUnet, scSEUnet, Unet_Xception_ResNetBlock, in keras

  2.13.13. [Efficient-Transformer](https://github.com/zyxu1996/Efficient-Transformer) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/18/3585): Efficient Transformer for Remote Sensing Image Segmentation

  2.13.14. [weakly_supervised](https://github.com/LobellLab/weakly_supervised) -> code for the 2020 [paper](https://www.mdpi.com/2072-4292/12/2/207): Weakly Supervised Deep Learning for Segmentation of Remote Sensing Imagery

  2.13.15. [HRCNet-High-Resolution-Context-Extraction-Network](https://github.com/zyxu1996/HRCNet-High-Resolution-Context-Extraction-Network) -> code to 2021 [paper](https://www.mdpi.com/2072-4292/13/1/71): High-Resolution Context Extraction Network for Semantic Segmentation of Remote Sensing Images

  2.13.16. [Semantic segmentation of SAR images using a self supervised technique](https://github.com/cattale93/pytorch_self_supervised_learning)

  2.13.17. [satellite-segmentation-pytorch](https://github.com/obravo7/satellite-segmentation-pytorch) -> explores a wide variety of image augmentations to increase training dataset size

  2.13.18. [IEEE_TGRS_SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer) -> code for 2021 [paper](https://arxiv.org/abs/2107.02988): Spectralformer: Rethinking hyperspectral image classification with transformers

  2.13.19. [Unsupervised Segmentation of Hyperspectral Remote Sensing Images with Superpixels](https://github.com/mpBarbato/Unsupervised-Segmentation-of-Hyperspectral-Remote-Sensing-Images-with-Superpixels) -> code for 2022 [paper](https://arxiv.org/abs/2204.12296)

  2.13.20. [Semantic-Segmentation-with-Sparse-Labels](https://github.com/Hua-YS/Semantic-Segmentation-with-Sparse-Labels) -> codes and data for learning from sparse annotations

  2.13.21. [SNDF](https://github.com/mi18/SNDF) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271619302606): Superpixel-enhanced deep neural forest for remote sensing image semantic segmentation

  2.13.22. [Satellite-Image-Classification](https://github.com/yxian29/Satellite-Image-Classification) -> using random forest or support vector machines (SVM) and sklearn

  2.13.23. [dynamic-rs-segmentation](https://github.com/keillernogueira/dynamic-rs-segmentation) -> code for 2019 [paper](https://arxiv.org/abs/1804.04020): Dynamic Multi-Context Segmentation of Remote Sensing Images based on Convolutional Networks

  2.13.24. [Remote-sensing-image-semantic-segmentation-tf2](https://github.com/TachibanaYoshino/Remote-sensing-image-semantic-segmentation-tf2) -> remote sensing image semantic segmentation repository based on tf.keras includes backbone networks such as resnet, densenet, mobilenet, and segmentation networks such as deeplabv3+, pspnet, panet, and refinenet

  2.13.25. [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) -> Segmentation models with pretrained backbones, has been used in multiple winning solutions to remote sensing competitions

  2.13.26. [SSRN](https://github.com/zilongzhong/SSRN) -> code for 2017 [paper](https://ieeexplore.ieee.org/document/8061020): Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework

  2.13.27. [SO-DNN](https://github.com/PanXinZebra/SO-DNN) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002525): Simplified object-based deep neural network for very high resolution remote sensing image classification

  2.13.28. [SANet](https://github.com/mrluin/SANet-PyTorch) -> code for 2019 [paper](https://arxiv.org/abs/1907.03089): Scale-Aware Network for Semantic Segmentation of High-Resolution Aerial Images

  2.13.29. [aerial-segmentation](https://github.com/alpemek/aerial-segmentation) -> code for 2017 [paper](https://arxiv.org/abs/1707.06879): Learning Aerial Image Segmentation from Online Maps

  2.13.30. [IterativeSegmentation](https://github.com/gaudetcj/IterativeSegmentation) -> code for 2016 [paper](https://arxiv.org/abs/1608.03440): Recurrent Neural Networks to Correct Satellite Image Classification Maps

  2.13.31. [Detectron2 FPN + PointRend Model for amazing Satellite Image Segmentation](https://affine.medium.com/detectron2-fpn-pointrend-model-for-amazing-satellite-image-segmentation-183456063e15) -> 15% increase in accuracy when compared to the U-Net model

  2.13.32. [HybridSN](https://github.com/gokriznastic/HybridSN) -> code for 2019 [paper](https://arxiv.org/abs/1902.06701): HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification. Also a [pytorch implementation here](https://github.com/purbayankar/HybridSN-pytorch)

  2.13.33. [TNNLS_2022_X-GPN](https://github.com/B-Xi/TNNLS_2022_X-GPN) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9740412): Semisupervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification

  2.13.34. [singleSceneSemSegTgrs2022](https://github.com/sudipansaha/singleSceneSemSegTgrs2022) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9773162): Unsupervised Single-Scene Semantic Segmentation for Earth Observation

  2.13.35. [A-Fast-and-Compact-3-D-CNN-for-HSIC](https://github.com/mahmad00/A-Fast-and-Compact-3-D-CNN-for-HSIC) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9307220): A Fast and Compact 3-D CNN for Hyperspectral Image Classification

  2.13.36. [HSNRS](https://github.com/Walkerlikesfish/HSNRS) -> code for 2017 [paper](https://www.mdpi.com/2072-4292/9/6/522): Hourglass-ShapeNetwork Based Semantic Segmentation for High Resolution Aerial Imagery

  2.13.37. [GiGCN](https://github.com/ShuGuoJ/GiGCN) -> code for 2022 [paper](https://pubmed.ncbi.nlm.nih.gov/35724277/): Graph-in-Graph Convolutional Network for Hyperspectral Image Classification

  2.13.38. [SSAN](https://github.com/EtPan/SSAN) -> code for 2019 [paper](https://www.mdpi.com/2072-4292/11/8/963): Spectral-Spatial Attention Networks for Hyperspectral Image Classification

  2.13.39. [drone-images-semantic-segmentation](https://github.com/ayushdabra/drone-images-semantic-segmentation) -> Multiclass Semantic Segmentation of Aerial Drone Images Using Deep Learning

  2.13.40. [Satellite-Image-Segmentation-with-Smooth-Blending](https://github.com/MaitrySinha21/Satellite-Image-Segmentation-with-Smooth-Blending) -> uses [Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)

  2.13.41. [BayesianUNet](https://github.com/tha-santacruz/BayesianUNet) -> Pytorch Bayesian UNet model for segmentation and uncertainty prediction, applied to the Potsdam Dataset

  2.13.42. [RAANet](https://github.com/Lrr0213/RAANet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/13/3109): RAANet: A Residual ASPP with Attention Framework for Semantic Segmentation of High-Resolution Remote Sensing Images

  2.13.43. [wheelRuts_semanticSegmentation](https://github.com/SmartForest-no/wheelRuts_semanticSegmentation) -> code for 2022 [paper](https://academic.oup.com/forestry/advance-article/doi/10.1093/forestry/cpac023/6627280): Mapping wheel-ruts from timber harvesting operations using deep learning techniques in drone imagery

  2.13.44. [LWN-for-UAVRSI](https://github.com/syliudf/LWN-for-UAVRSI) -> Light-Weight Semantic Segmentation Network for UAV Remote Sensing Images, applied to Vaihingen, UAVid and UDD6 datasets

  2.13.45. [hypernet](https://github.com/ESA-PhiLab/hypernet) -> library which implements; accurate hyperspectral image (HSI) segmentation and analysis using deep neural networks, optimization of deep neural network architectures for hyperspectral data segmentation, hyperspectral data augmentation, validation of existent and emerging HSI segmentation algorithms, simulation of multispectral data using HSI

  2.13.46. [ST-UNet](https://github.com/XinnHe/ST-UNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9686686): Swin Transformer Embedding UNet for Remote Sensing Image Semantic Segmentation

  2.13.47. [EDFT](https://github.com/h1063135843/EDFT) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/5/1294): Efficient Depth Fusion Transformer for Aerial Image Semantic Segmentation

  2.13.48. [WiCoNet](https://github.com/ggsDing/WiCoNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9759447): Looking Outside the Window: Wide-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images

  2.13.49. [CRGNet](https://github.com/YonghaoXu/CRGNet) -> code for 2022 [paper](https://arxiv.org/abs/2202.03740): Consistency-Regularized Region-Growing Network for Semantic Segmentation of Urban Scenes with Point-Level Annotations

  2.13.50. [SA-UNet](https://github.com/Yancccccc/SA-UNet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/15/3591): Improved U-Net Remote Sensing Classification Algorithm Fusing Attention and Multiscale Features

  2.13.51. [MANet](https://github.com/lironui/Multi-Attention-Network) -> code for 2020 [paper](https://arxiv.org/abs/2009.02130): Multi-Attention-Network for Semantic Segmentation of Fine Resolution Remote Sensing Images

  2.13.52. [BANet](https://github.com/lironui/BANet) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/16/3065): Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images

  2.13.53. [MACU-Net](https://github.com/lironui/MACU-Net) -> code for 2022 [paper](https://arxiv.org/abs/2007.13083): MACU-Net for Semantic Segmentation of Fine-Resolution Remotely Sensed Images

  2.13.54. [DNAS](https://github.com/faye0078/DNAS) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/16/3864): DNAS: Decoupling Neural Architecture Search for High-Resolution Remote Sensing Image Semantic Segmentation

  2.13.55. [A2-FPN](https://github.com/lironui/A2-FPN) -> code for 2021 [paper](https://arxiv.org/abs/2102.07997): A2-FPN for Semantic Segmentation of Fine-Resolution Remotely Sensed Images

  2.13.56. [MAResU-Net](https://github.com/lironui/MAResU-Net) -> code for 2020 [paper](https://arxiv.org/abs/2011.14302): Multi-stage Attention ResU-Net for Semantic Segmentation of Fine-Resolution Remote Sensing Images

  2.13.57. [ml_segmentation](https://github.com/dgriffiths3/ml_segmentation) -> semantic segmentation of buildings using Random Forest, Support Vector Machine (SVM) & Gradient Boosting Classifier (GBC)

  2.13.58. [RSEN](https://github.com/YonghaoXu/RSEN) -> code for 2021 [paper](https://arxiv.org/abs/2104.03765): Robust Self-Ensembling Network for Hyperspectral Image Classification

  2.13.59. [MSNet](https://github.com/taochx/MSNet) -> code for 2022 [paper](https://www.tandfonline.com/doi/full/10.1080/15481603.2022.2101728): MSNet: multispectral semantic segmentation network for remote sensing images

  2.13.60. [k-textures](https://zenodo.org/record/6359859#.Yytt6OzMK3I) -> code (R) for 2022 [paper](https://www.frontiersin.org/articles/10.3389/fenvs.2022.946729/full): K-textures, a self-supervised hard clustering deep learning algorithm for satellite image segmentation

  2.13.61. [Swin-Transformer-Semantic-Segmentation](https://github.com/koechslin/Swin-Transformer-Semantic-Segmentation) -> code for 2021 [paper](https://arxiv.org/abs/2110.05812): Satellite Image Semantic Segmentation

  2.13.62. [UDA_for_RS](https://github.com/Levantespot/UDA_for_RS) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/19/4942): Unsupervised Domain Adaptation for Remote Sensing Semantic Segmentation with Transformer

  2.13.63. [A-3D-CNN-AM-DSC-model-for-hyperspectral-image-classification](https://github.com/hahatongxue/A-3D-CNN-AM-DSC-model-for-hyperspectral-image-classification) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/9/2215): Attention Mechanism and Depthwise Separable Convolution Aided 3DCNN for Hyperspectral Remote Sensing Image Classification

  2.13.64. [contrastive-distillation](https://github.com/edornd/contrastive-distillation) -> code for [paper](https://arxiv.org/abs/2112.03814): A Contrastive Distillation Approach for Incremental Semantic Segmentation in Aerial Images

  2.13.65. [SegForestNet](https://github.com/gritzner/SegForestNet) -> code for 2023 paper: SegForestNet: Spatial-Partitioning-Based Aerial Image Segmentation

  2.13.66. [MFVNet](https://github.com/weichenrs/MFVNet) -> code for 2023 paper: MFVNet: Deep Adaptive Fusion Network with Multiple Field-of-Views for Remote Sensing Image Semantic Segmentation

  2.13.67. [Wildebeest-UNet](https://github.com/zijing-w/Wildebeest-UNet) -> detecting wildebeest and zebras in Serengeti-Mara ecosystem from very-high-resolution satellite imagery

  2.13.68. [segment-anything-eo](https://github.com/aliaksandr960/segment-anything-eo) -> Earth observation tools for Meta AI Segment Anything (SAM - Segment Anything Model)

  2.13.69. [HR-Image-classification_SDF2N](https://github.com/SicongLiuRS/HR-Image-classification_SDF2N) -> code for 2023 paper: A Shallow-to-Deep Feature Fusion Network for VHR Remote Sensing Image Classification

  2.13.70. [TDD](https://github.com/Jingtao-Li-CVer/TDD) -> code for 2023 paper: One-Step Detection Paradigm for Hyperspectral Anomaly Detection via Spectral Deviation Relationship Learning

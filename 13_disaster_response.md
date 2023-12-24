# 13. Disaster response

<p align="center">
  <img src="images/disaster.png" width="750">
  <br>
  <b>Detecting buildings destroyed in a disaster.</b>
</p>

Remote sensing images are used in disaster response to identify and assess damage to an area. This imagery can be used to detect buildings that are damaged or destroyed, identify roads and road networks that are blocked, determine the size and shape of a disaster area, and identify areas that are at risk of flooding. Remote sensing images can also be used to detect and monitor the spread of forest fires and monitor vegetation health. Also checkout the sections on change detection and water/fire/building segmentation. [Image source](https://developer.nvidia.com/blog/ai-helps-detect-disaster-damage-from-satellite-imagery/).

  13.1. [DisaVu](https://github.com/SrzStephen/DisaVu) -> combines building & damage detection and provides an app for viewing predictions

  13.2. [Soteria](https://github.com/Soteria-ai/Soteria) -> uses machine learning with satellite imagery to map natural disaster impacts for faster emergency response

  13.3. [DisasterHack](https://github.com/MarjorieRWillner/DisasterHack) -> Wildfire Mitigation: Computer Vision Identification of Hazard Fuels Using Landsat

  13.4. [forestcasting](https://github.com/ivanzvonkov/forestcasting) -> Forest fire prediction powered by analytics

  13.5. [Machine Learning-based Damage Assessment for Disaster Relief on Google AI blog](https://ai.googleblog.com/2020/06/machine-learning-based-damage.html) -> uses object detection to locate buildings, then a classifier to determine if a building is damaged. Challenge of generalising due to small dataset

  13.6. [hurricane_damage](https://github.com/allankapoor/hurricane_damage) -> Post-hurricane structure damage assessment based on aerial imagery with CNN

  13.7. [rescue](https://github.com/dbdmg/rescue) -> code of the paper: Attention to fires: multi-channel deep-learning models forwildfire severity prediction

  13.8. [Disaster-Classification](https://github.com/bostankhan6/Disaster-Classification) -> A disaster classification model to predict the type of disaster given an input image

  13.9. [Coarse-to-fine weakly supervised learning method for green plastic cover segmentation](https://github.com/lauraset/Coarse-to-fine-weakly-supervised-GPC-segmentation) -> with [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271622001095)

  13.10. [Detection of destruction in satellite imagery](https://github.com/usmanali414/Destruction-Detection-in-Satellite-Imagery)

  13.11. [BDD-Net](https://github.com/jinyuan30/Recognize-damaged-buildings) -> code for 2020 paper: A General Protocol for Mapping Buildings Damaged by a Wide Range of Disasters Based on Satellite Imagery

  13.12. [building-segmentation-disaster-resilience](https://github.com/kbrodt/building-segmentation-disaster-resilience) -> 2nd place solution in the Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience

  13.13. [Flooding Damage Detection from Post-Hurricane Satellite Imagery Based on Convolutional Neural Networks](https://github.com/weining20000/Flooding-Damage-Detection-from-Post-Hurricane-Satellite-Imagery-Based-on-CNN)

  13.14. [IBM-Disaster-Response-Hack](https://github.com/NicoDeshler/IBM-Disaster-Response-Hack) -> identifying optimal terrestrial routes through calamity-stricken areas. Satellite image data informs road condition assessment and obstruction detection

  13.15. [Automatic Damage Annotation on Post-Hurricane Satellite Imagery](https://dds-lab.github.io/disaster-damage-detection/) -> detect damaged buildings using tensorflow object detection API. With repos [here](https://github.com/DDS-Lab/disaster-image-processing) and [here](https://github.com/annieyan/PreprocessSatelliteImagery-ObjectDetection)

  13.16. [Hurricane-Damage-Detection](https://github.com/Ryan-Awad/Hurricane-Damage-Detection) -> Waterloo's Hack the North 2020++ submission. A convolutional neural network model used to detect hurricane damage in RGB satellite images

  13.17. [wildfire_forecasting](https://github.com/Orion-AI-Lab/wildfire_forecasting) -> code for 2021 [paper](https://arxiv.org/abs/2111.02736): Deep Learning Methods for Daily Wildfire Danger Forecasting. Uses ConvLSTM

  13.18. [Satellite Image Analysis with fast.ai for Disaster Recovery](https://appsilon.com/satellite-image-analysis-with-fast-ai-for-disaster-recovery/)

  13.19. [shackleton](https://github.com/avanetten/shackleton) -> leverages remote sensing imagery and machine learning techniques to provide insights into various transportation and evacuation scenarios in an interactive dashboard that conducts real-time computation

  13.20. [ai-vegetation-fuel](https://github.com/ecmwf-projects/ai-vegetation-fuel) -> Predicting Fuel Load from earth observation data using Machine Learning, using LightGBM & CatBoost

  13.21. [AI Helps Detect Disaster Damage From Satellite Imagery](https://developer.nvidia.com/blog/ai-helps-detect-disaster-damage-from-satellite-imagery/) -> NVIDIA blog post

  13.22. [Turkey-Earthquake-2023-Building-Change-Detection](https://github.com/blackshark-ai/Turkey-Earthquake-2023-Building-Change-Detection) -> The repository contains building footprints derived from Maxar open data imagery and change detection results by blackshark-ai

  13.23. [MS4D-Net-Building-Damage-Assessment](https://github.com/YJ-He/MS4D-Net-Building-Damage-Assessment) -> code for 2022 paper: MS4D-Net: Multitask-Based Semi-Supervised Semantic Segmentation Framework with Perturbed Dual Mean Teachers for Building Damage Assessment from High-Resolution Remote Sensing Imagery

  13.24. [DAHiTra](https://github.com/nka77/DAHiTra) -> code for 2022 [paper](https://arxiv.org/abs/2208.02205): Large-scale Building Damage Assessment using a Novel Hierarchical Transformer Architecture on Satellite Images. Uses xView2 xBD dataset
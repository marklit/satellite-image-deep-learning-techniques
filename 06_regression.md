# 6. Regression

<p align="center">
  <img src="images/regression.png" width="300">
  <br>
  <b>Regression prediction of windspeed.</b>
</p>

Regression in remote sensing involves predicting continuous variables such as wind speed, tree height, or soil moisture from an image. Both classical machine learning and deep learning approaches can be used to accomplish this task. Classical machine learning utilizes feature engineering to extract numerical values from the input data, which are then used as input for a regression algorithm like linear regression. On the other hand, deep learning typically employs a convolutional neural network (CNN) to process the image data, followed by a fully connected neural network (FCNN) for regression. The FCNN is trained to map the input image to the desired output, providing predictions for the continuous variables of interest. [Image source](https://github.com/h-fuzzy-logic/python-windspeed)

  6.1. [python-windspeed](https://github.com/h-fuzzy-logic/python-windspeed) -> Predicting windspeed of hurricanes from satellite images, uses CNN regression in keras

  6.2. [hurricane-wind-speed-cnn](https://github.com/23ccozad/hurricane-wind-speed-cnn) -> Predicting windspeed of hurricanes from satellite images, uses CNN regression in keras

  6.3. [GEDI-BDL](https://github.com/langnico/GEDI-BDL) -> code for paper: Global canopy height regression and uncertainty estimation from GEDI LIDAR waveforms with deep ensembles

  6.4. [Traffic density estimation as a regression problem instead of object detection](https://omdena.com/blog/ai-road-safety/) -> inspired by paper: Traffic density estimation method from small satellite imagery: Towards frequent remote sensing of car traffic

  6.5. [OpticalWaveGauging_DNN](https://github.com/OpticalWaveGauging/OpticalWaveGauging_DNN) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0378383919301243): Optical wave gauging using deep neural networks

  6.6. [satellite-pose-estimation](https://github.com/eio/satellite-pose-estimation) -> adapts a ResNet50 model architecture to perform pose estimation on several series of satellite images (both real and synthetic)

  6.7. [Tropical Cyclone Wind Estimation Competition](https://mlhub.earth/10.34911/rdnt.xs53up) -> on RadiantEarth MLHub

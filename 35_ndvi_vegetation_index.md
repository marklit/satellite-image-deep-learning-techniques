# 35. NDVI - vegetation index

Normalized Difference Vegetation Index (NDVI) is an index used to measure the amount of healthy vegetation in a given area. It is calculated by taking the difference between the near-infrared (NIR) and red (red) bands of a satellite image, and dividing by the sum of the two bands. NDVI can be used to identify areas of healthy vegetation and to assess the health of vegetation in a given area.

  35.1. Calculated via band math `ndvi = np.true_divide((ir - r), (ir + r))` but challenging due to the size of the imagery

  35.2. [Example notebook local](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/ndvi_calculation.ipynb)

  35.3. [Landsat data in cloud optimised (COG) format analysed for NDVI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article here](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).

  35.4. [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”

  35.5. [Seeing Through the Clouds - Predicting Vegetation Indices Using SAR](https://medium.com/descarteslabs-team/seeing-through-the-clouds-34a24f84b599)

  35.6. [A walkthrough on calculating NDWI water index for flooded areas](https://towardsdatascience.com/how-to-compute-satellite-image-statistics-and-use-it-in-pandas-81864a489144) -> Derive zonal statistics from Sentinel 2 images using Rasterio and Geopandas

  35.7. [NDVI-Net](https://github.com/HaoZhang1018/NDVI-Net) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271620302185): NDVI-Net: A fusion network for generating high-resolution normalized difference vegetation index in remote sensing

  35.8. [Awesome-Vegetation-Index](https://github.com/px39n/Awesome-Vegetation-Index)

  35.9. [Remote-Sensing-Indices-Derivation-Tool](https://github.com/rander38/Remote-Sensing-Indices-Derivation-Tool) -> Calculate spectral remote sensing indices from satellite imagery

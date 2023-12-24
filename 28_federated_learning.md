# 28. Federated learning

Federated learning is an approach to distributed machine learning where a central processor coordinates the training of an individual model in each of its clients. It is a type of distributed ML which means that the data is distributed among different devices or locations and the model is trained on all of them. The central processor aggregates the model updates from all the clients and then sends the global model parameters back to the clients. This is done to protect the privacy of data, as the data remains on the local device and only the global model parameters are shared with the central processor. This technique can be used to train models with large datasets that cannot be stored in a single device, as well as to enable certain privacy-preserving applications.

  28.1. [Federated-Learning-for-Remote-Sensing](https://github.com/anandcu3/Federated-Learning-for-Remote-Sensing) ->  implementation of three Federated Learning models

  28.2. [Semantic-Segmentation-UNet-Federated](https://github.com/PratikGarai/Semantic-Segmentation-UNet-Federated) -> code for paper: FedUKD: Federated UNet Model with Knowledge Distillation for Land Use Classification from Satellite and Street Views

  28.3. [MM-FL](https://git.tu-berlin.de/rsim/MM-FL) -> code for paper: Learning Across Decentralized Multi-Modal Remote Sensing Archives with Federated Learning

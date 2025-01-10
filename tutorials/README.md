# Tutorials
These tutorials require the [IJava](https://github.com/SpencerPark/IJava) Jupyter notebook kernel, and Java 17+.

The tutorials expect the data and required jars to be in the same directory as the notebooks. The dataset download
links are given in the tutorial, and Tribuo's jars are on Maven Central, attached to the GitHub release, or you
can build it yourself with `mvn clean package -Pwith-deps-jar` using Apache Maven.

The tutorials cover:
- [Intro classification with Irises](irises-tribuo-v4.ipynb)
- [Intro regression with wine-quality](regression-tribuo-v4.ipynb)
- [Configuration files, provenance and feature transformations on MNIST](configuration-tribuo-v4.ipynb)
- [Clustering with K-Means](clustering-tribuo-v4.ipynb)
- [Clustering with HDBSCAN\*](clustering-hdbscan-tribuo-v4.ipynb)
- [Anomaly Detection with LibSVM](anomaly-tribuo-v4.ipynb)
- [Multi-label classification with Classifier Chains](multi-label-tribuo-v4.ipynb)
- [Loading columnar data](columnar-tribuo-v4.ipynb)
- [Document classification and extracting features from text](document-classification-tribuo-v4.ipynb)
- [Importing third-party models](external-models-tribuo-v4.ipynb)
- [Training and deploying TensorFlow models](tensorflow-tribuo-v4.ipynb)
- [ONNX export and deployment](onnx-export-tribuo-v4.ipynb)
- [Model reproducibility](reproducibility-tribuo-v4.ipynb)
- [Documenting Tribuo Models with Model Cards](modelcard-tribuo-v4.ipynb)
- [Feature selection for classification problems](feature-selection-tribuo-v4.ipynb)

# Tutorials

These tutorials require the [IJava](https://github.com/SpencerPark/IJava) Jupyter notebook kernel, and Java 10+.

The tutorials expect the data and required jars to be in the same directory as the notebooks. The dataset download
links are given in the tutorial, and Tribuo's jars are on Maven Central, attached to the GitHub release, or you
can build it yourself with `mvn clean package` using Apache Maven.
The code in them should work on Java 8 & 9 with the addition of types to replace the use of the `var` keyword
added in Java 10.

The tutorials cover:
- [Intro classification with Irises](irises-tribuo-v4.ipynb)
- [Intro regression with wine-quality](regression-tribuo-v4.ipynb)
- [Configuration files, provenance and feature transformations on MNIST](configuration-tribuo-v4.ipynb)
- [Clustering with K-Means](clustering-tribuo-v4.ipynb)
- [Anomaly Detection with LibSVM](anomaly-tribuo-v4.ipynb)
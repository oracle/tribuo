# Tribuo v4.2 Release Notes

Tribuo 4.2 adds new models, ONNX export for several types of models, a
reproducibility framework for recreating Tribuo models, easy deployment of
Tribuo models on Oracle Cloud, along with several smaller improvements and bug
fixes. We've added more tutorials covering the new features along with
multi-label classification, and further expanded the javadoc to cover all
public methods.

In Tribuo 4.1.0 and earlier there is a severe bug in multi-dimensional
regression models (i.e., regression tasks with multiple output dimensions).
Models other than `LinearSGDModel` and `SparseLinearModel` (apart from when
using the `ElasticNetCDTrainer`) have a bug in how the output dimension indices
are constructed, and may produce incorrect outputs for all dimensions (as the
output will be for a different dimension than the one named in the `Regressor`
object). This has been fixed, and loading in models trained in earlier versions
of Tribuo will patch the model to rearrange the dimensions appropriately.
Unfortunately this fix cannot be applied to tree based models, and so all
multi-output regression tree based models should be retrained using Tribuo 4.2
as they are irretrievably corrupt. Additionally when using standardization in
multi-output regression LibSVM models dimensions past the first dimension have
the model improperly stored and will also need to be retrained with Tribuo 4.2.
See [#177](https://github.com/oracle/tribuo/pull/177) for more details.

Note the KMeans implementation had several internal changes to support running
with a `java.lang.SecurityManager` which will break any subclasses of `KMeansTrainer`.
In most cases changing the signature of any overridden `mStep` method to match
the new signature, and allowing the `fjp` argument to be null in single threaded 
execution will fix the subclass.

## New models

In this release we've added [Factorization
Machines](https://www.computer.org/csdl/proceedings-article/icdm/2010/4256a995/12OmNwMFMfl),
[Classifier
Chains](https://link.springer.com/content/pdf/10.1007/s10994-011-5256-5.pdf)
and
[HDBSCAN\*](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14).
Factorization machines are a powerful non-linear predictor which uses a
factorized approximation to learn a per output feature-feature interaction term
in addition to a linear model. We've added Factorization Machines for
multi-class classification, multi-label classification and regression.
Classifier chains are an ensemble approach to multi-label classification which
given a specific ordering of the labels learns a chain of classifiers where
each classifier gets the features along with the predicted labels from earlier
in the chain. We also added ensembles of randomly ordered classifier chains
which work well in situations when the ground truth label ordering is unknown
(i.e., most of the time).  HDBSCAN is a hierarchical density based clustering
algorithm which chooses the number of clusters based on properties of the data
rather than as a hyperparameter. The Tribuo implementation can cluster a
dataset, and then at prediction time it provides the cluster the given
datapoint would be in without modifying the cluster structure.

- Classifier Chains ([#149](https://github.com/oracle/tribuo/pull/149)), which
  also adds the jaccard score as a multi-label evaluation metric, and a
multi-label voting combiner for use in multi-label ensembles.
- Factorization machines ([#179](https://github.com/oracle/tribuo/pull/179)).
- HDBSCAN ([#196](https://github.com/oracle/tribuo/pull/196)).

## ONNX Export

The [ONNX](https://onnx.ai) format is a cross-platform and cross-library model
exchange format. Tribuo can already serve ONNX models via its [ONNX
Runtime](https://onnxruntime.ai) interface, and now has the ability to export
models in ONNX format for serving on edge devices, in cloud services, or in
other languages like Python or C#.

In this release Tribuo supports exporting linear models (multi-class
classification, multi-label classification and regression), sparse linear
regression models, factorization machines (multi-class classification,
multi-label classification and regression), LibLinear models (multi-class
classification and regression), LibSVM models (multi-class classification and
regression), along with ensembles of those models, including arbitrary levels
of ensemble nesting. We plan to expand this coverage to more models over time,
however for TensorFlow we recommend users export those models as a Saved Model
and use the Python tf2onnx converter.

Tribuo models exported in ONNX format preserve their provenance information in
a metadata field which is accessible when the ONNX model is loaded back into
Tribuo. The provenance is stored as a protobuf so could be read from other
libraries or platforms if necessary.

The ONNX export support is in a separate module with no dependencies, and could
be used elsewhere on the JVM to support generating ONNX graphs. We welcome
contributions to build out the ONNX support in that module.

- ONNX export for LinearSGDModels
  ([#154](https://github.com/oracle/tribuo/pull/154)), which also adds a
multi-label output transformer for scoring multi-label ONNX models.
- ONNX export for SparseLinearModel ([#163](https://github.com/oracle/tribuo/pull/163)).
- Add provenance to ONNX exported models ([#182](https://github.com/oracle/tribuo/pull/182)).
- Refactor ONNX tensor creation ([#187](https://github.com/oracle/tribuo/pull/187)).
- ONNX ensemble export support ([#186](https://github.com/oracle/tribuo/pull/186)).
- ONNX export for LibSVM and LibLinear ([#191](https://github.com/oracle/tribuo/pull/191)).
- Refactor ONNX support to improve type safety ([#199](https://github.com/oracle/tribuo/pull/199)).
- Extract ONNX support into separate module ([#TBD](https://github.com/oracle/tribuo/pull/)).

## Reproducibility Framework

Tribuo has strong model metadata support via its provenance system which
records how models, datasets and evaluations are created. In this release we
enhance this support by adding a push-button reproduction framework which
accepts either a model provenance or a model object and rebuilds the complete
training pipeline, ensuring consistent usage of RNGs and other mutable state.

This allows Tribuo to easily rebuild models to see if updated datasets could
change performance, or even if the model is actually reproducible (which may be
required for regulatory reasons).  Over time we hope to expand this support
into a full experimental framework, allowing models to be rebuilt with
hyperparameter or data changes as part of the data science process or for
debugging models in production.

This framework was written by Joseph Wonsil and Prof. Margo Seltzer at the
University of British Columbia as part of a collaboration between Prof. Seltzer
and Oracle Labs. We're excited to continue working with Joe, Margo and the rest
of the lab at UBC, as this is excellent work.

Note the reproducibility framework module requires Java 16 or greater, and is
thus not included in the `tribuo-all` meta-module.

- Reproducibility framework ([#185](https://github.com/oracle/tribuo/pull/185), with minor changes in [#189](https://github.com/oracle/tribuo/pull/189) and [#190](https://github.com/oracle/tribuo/pull/190)).

## OCI Data Science Integration

[Oracle Cloud Data
Science](https://www.oracle.com/data-science/cloud-infrastructure-data-science.html)
is a platform for building and deploying models in Oracle Cloud.  The model
deployment functionality wraps a Python runtime and deploys them with an
auto-scaler at a REST endpoint. In this release we've added support for
deploying Tribuo models which are ONNX exportable directly to OCI DS, allowing
scale-out deployments of models from the JVM. We also added a `OCIModel`
wrapper which scores Tribuo `Example` objects using a deployed model's REST
endpoint, allowing easy use of cloud resources for ML on the JVM.

- Oracle Cloud Data Science integration ([#200](https://github.com/oracle/tribuo/pull/200)).

## Small improvements

- Date field processor and locale support in metadata extractors ([#148](https://github.com/oracle/tribuo/pull/148))
- Multi-output response processor allowing loading different formats of multi-label and multi-dimensional regression datasets ([#150](https://github.com/oracle/tribuo/pull/150))
- ARM dev profile for compiling Tribuo on ARM platforms ([#152](https://github.com/oracle/tribuo/pull/152))
- Refactor CSVLoader so it uses CSVDataSource and parses CSV files using RowProcessor, allowing an easy transition to more complex columnar extraction ([#153](https://github.com/oracle/tribuo/pull/153))
- Configurable anomaly demo data source ([#160](https://github.com/oracle/tribuo/pull/160))
- Configurable clustering demo data source ([#161](https://github.com/oracle/tribuo/pull/161))
- Configurable classification demo data source ([#162](https://github.com/oracle/tribuo/pull/162)) 
- Multi-Label tutorial and configurable multi-label demo data source ([#166](https://github.com/oracle/tribuo/pull/166)) (also adds a multi-label tutorial) plus fix in [#168](https://github.com/oracle/tribuo/pull/168) after #167
- Add javadoc for all public methods and fields ([#175](https://github.com/oracle/tribuo/pull/175)) (also fixes a bug in Util.vectorNorm)
- Add hooks for model equality checks to trees and LibSVM models ([#183](https://github.com/oracle/tribuo/pull/183)) (also fixes a bug in liblinear get top features)
- XGBoost 1.5.0 ([#192](https://github.com/oracle/tribuo/pull/192))
- TensorFlow Java 0.4.0 ([#195](https://github.com/oracle/tribuo/pull/195)) (note this changes Tribuo's TF API slightly as TF-Java 0.4.0 has a different method of initializing the session)
- KMeans now uses dense vectors when appropriate, speeding up training ([#201](https://github.com/oracle/tribuo/pull/201))
- Documentation updates, ONNX and reproducibility tutorials ([#205](https://github.com/oracle/tribuo/pull/205))

## Bug fixes

- NPE fix for LIME explanations using models which don't support per class weights ([#157](https://github.com/oracle/tribuo/pull/157))
- Fixing a bug in multi-label evaluation which swapped FP for FN ([#167](https://github.com/oracle/tribuo/pull/167))
- Persist CSVDataSource headers in the provenance ([#171](https://github.com/oracle/tribuo/pull/171))
- Fixing LibSVM and LibLinear so they have reproducible behaviour ([#172](https://github.com/oracle/tribuo/pull/172))
- Provenance fix for TransformTrainer and an extra factory for XGBoostExternalModel so you can make them from an in memory booster ([#176](https://github.com/oracle/tribuo/pull/176))
- Fix multidimensional regression ([#177](https://github.com/oracle/tribuo/pull/177)) (fixes regression ids, fixes libsvm so it emits correct standardized models, adds support for per dimension feature weights in XGBoostRegressionModel)
- Fix provenance generation for FieldResponseProcessor and BinaryResponseProcessor ([#178](https://github.com/oracle/tribuo/pull/178)) 
- Normalize LibSVMDataSource paths consistently in the provenance ([#181](https://github.com/oracle/tribuo/pull/181))
- KMeans and KNN now run correctly when using OpenSearch's SecurityManager ([#197](https://github.com/oracle/tribuo/pull/197))

## Contributors

- Adam Pocock ([@Craigacp](https://github.com/Craigacp))
- Jack Sullivan ([@JackSullivan](https://github.com/JackSullivan))
- Joseph Wonsil ([@jwons](https://github.com/jwons))
- Philip Ogren ([@pogren](https://github.com/pogren))
- Jeffrey Alexander ([@jhalexand](https://github.com/jhalexand))
- Geoff Stewart ([@geoffreydstewart](https://github.com/geoffreydstewart))


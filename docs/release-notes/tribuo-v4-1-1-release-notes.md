# Tribuo v4.1.1 Release Notes

This is the first patch release for Tribuo v4.1. The main fixes in this release
are to the multi-dimensional output regression support, and to support the use
of KMeans and KNN models when running under a restrictive `SecurityManager`.
Additionally this release pulls in TensorFlow-Java 0.4.0 which upgrades the
TensorFlow native library to 2.7.0 fixing several CVEs. Note those CVEs may not
be applicable to TensorFlow-Java, as many of them relate to Python codepaths
which are not included in TensorFlow-Java. Note the TensorFlow upgrade is a
breaking API change as graph initialization is handled differently in this
release, which causes unavoidable changes in Tribuo's TF API.

## Multi-dimensional Regression fix

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

## Bug fixes

- NPE fix for LIME explanations using models which don't support per class weights ([#157](https://github.com/oracle/tribuo/pull/157)).
- Fixing a bug in multi-label evaluation which swapped FP for FN ([#167](https://github.com/oracle/tribuo/pull/167)).
- Fixing LibSVM and LibLinear so they have reproducible behaviour ([#172](https://github.com/oracle/tribuo/pull/172)).
- Provenance fix for TransformTrainer and an extra factory for XGBoostExternalModel so you can make them from an in memory booster ([#176](https://github.com/oracle/tribuo/pull/176))
- Fix multidimensional regression ([#177](https://github.com/oracle/tribuo/pull/177)) (fixes regression ids, fixes libsvm so it emits correct standardized models, adds support for per dimension feature weights in XGBoostRegressionModel).
- Normalize LibSVMDataSource paths consistently in the provenance ([#181](https://github.com/oracle/tribuo/pull/181)).
- KMeans and KNN now run correctly when using OpenSearch's SecurityManager ([#197](https://github.com/oracle/tribuo/pull/197)).
- TensorFlow-Java 0.4.0 ([#195](https://github.com/oracle/tribuo/pull/195)).


## Contributors

- Adam Pocock ([@Craigacp](https://github.com/Craigacp))
- Jack Sullivan ([@JackSullivan](https://github.com/JackSullivan))
- Philip Ogren ([@pogren](https://github.com/pogren))
- Jeffrey Alexander ([@jhalexand](https://github.com/jhalexand))


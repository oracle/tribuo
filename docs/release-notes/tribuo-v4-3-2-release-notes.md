# Tribuo v4.3.2 Release Notes

Patch release to bring many dependencies up to the latest version, and integrate various small fixes. This release has protobuf support for all the main classes (two were accidentally missed from the earlier 4.3 releases), along with protobuf serialization tests. Tribuo v5 will remove support for `java.io.Serializable` and require protobuf serialization.

## Updates
* Bumping OCI Java SDK to 3.48.0, junit to 5.11.3, Jackson to 2.18.0, OpenCSV to 5.9 by @craigacp in https://github.com/oracle/tribuo/pull/380
* Bumping to protobuf-java 3.25.6 and regenerating all the protobufs by @craigacp in https://github.com/oracle/tribuo/pull/381 and https://github.com/oracle/tribuo/pull/395
* Moving to TF-Java 1.0.0 by @craigacp in https://github.com/oracle/tribuo/pull/367. Note the TensorFlow interface requires Java 11, as TF-Java v1.0 requires Java 11.
* Moving to OLCUT 5.3.1 by @craigacp in https://github.com/oracle/tribuo/pull/387
* Moving to ONNX Runtime 1.20.0 by @craigacp in https://github.com/oracle/tribuo/pull/368

## Bug fixes
* Improve the determination of cluster exemplars by @geoffreydstewart in https://github.com/oracle/tribuo/pull/356
* Dataset.createTransformers fix for DatasetView/TransformTrainer by @craigacp in https://github.com/oracle/tribuo/pull/364
* Fix `SQLDataSource` connection leak by @JackSullivan in https://github.com/oracle/tribuo/pull/376
* Fixing a multithreading bug in WordpieceTokenizer by @craigacp in https://github.com/oracle/tribuo/pull/382
* Fixing a bug in IntArrayContainer.merge and adding tests by @craigacp in https://github.com/oracle/tribuo/pull/384
* Matrix Factorization determinant calculation & SparseVector.subtract fixes by @craigacp taken from https://github.com/oracle/tribuo/pull/369
* Tag support for TensorFlowSavedModelExternalModel by @craigacp in https://github.com/oracle/tribuo/pull/393

## Protobuf serialization fixes & tests
* Add deserialization tests for 4.3 protobufs in AnomalyDetection, Clustering, MultiLabel by @craigacp in https://github.com/oracle/tribuo/pull/318
* Adding protobuf serialization for TransformedModel and IndependentSequenceModel by @craigacp in https://github.com/oracle/tribuo/pull/321
* Add deserialization tests for 4.3 protobufs in Regression by @craigacp in https://github.com/oracle/tribuo/pull/322
* Fixes for protobuf creation in a few classes by @craigacp in https://github.com/oracle/tribuo/pull/323
* Add deserialization tests for 4.3 protobufs in Classification by @craigacp in https://github.com/oracle/tribuo/pull/345
* Add deserialization tests for 4.3 protobufs in Math by @craigacp and @pogren in https://github.com/oracle/tribuo/pull/346
* Add deserialization tests for 4.3 protobufs in Core by @craigacp in https://github.com/oracle/tribuo/pull/386

## Contributors

- Adam Pocock ([@Craigacp](https://github.com/Craigacp))
- Jeffrey Alexander ([@jhalexand](https://github.com/jhalexand))
- Jack Sullivan ([@JackSullivan](https://github.com/JackSullivan))
- Philip Ogren ([@pogren](https://github.com/pogren))
- Geoff Stewart ([@geoffreydstewart](https://github.com/geoffreydstewart))

**Full Changelog**: https://github.com/oracle/tribuo/compare/v4.3.1...v.4.3.2

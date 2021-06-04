# Tribuo v4.0.2 Release Notes

This is the first Tribuo point release after the initial public announcement.
It fixes many of the issues our early users have found, and improves the
documentation in the areas flagged by those users. We also added a couple of
small new methods as part of fixing the bugs, and added two new tutorials: one
on columnar data loading and one on external model loading (i.e. XGBoost and
ONNX models).

Bugs fixed:
- Fixed a locale issue in the evaluation tests.
- Fixed issues with RowProcessor (expand regexes not being called, improper provenance capture).
- `IDXDataSource` now throws `FileNotFoundException` rather than a mysterious `NullPointerException` when it can't find the file.
- Fixed issues in `JsonDataSource` (consistent exceptions thrown, proper termination of reading in several cases).
- Fixed an issue where regression models couldn't be serialized due to a non-serializable lambda.
- Fixed UTF-8 BOM issues in CSV loading.
- Fixed an issue where `LibSVMTrainer` didn't track state between repeated calls to train.
- Fixed issues in the evaluators to ensure consistent exception throwing when discovering unlabelled or unknown ground truth outputs.
- Fixed a bug in ONNX `LabelTransformer` where it wouldn't read pytorch outputs properly.
- Bumped to OLCUT 5.1.5 to fix a provenance -> configuration conversion issue.

New additions:
- Added a method which converts a Jackson `ObjectNode` into a `Map` suitable for the `RowProcessor`.
- Added missing serialization tests to all the models.
- Added a getInnerModels method to `LibSVMModel`, `LibLinearModel` and `XGBoostModel` to allow users to access a copy of the internal models.
- More documentation.
- Columnar data loading tutorial.
- External model (XGBoost & ONNX) tutorial.

Dependency updates:
- OLCUT 5.1.5 (brings in jline 3.16.0 and jackson 2.11.3).

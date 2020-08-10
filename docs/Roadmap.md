# Roadmap

This is the list of features we are considering for the next few feature
releases of Tribuo.  As always, we're interested in the community's feedback,
and Tribuo's development will be informed by what our users want.

Tribuo conforms to semantic versioning, so some of these features may be held
back until the next major release, while others will land in minor releases.

We've broken this list up into several categories: API changes, internals, new
ML algorithms, performance and documentation.

## API Changes

- Support for ranking and other structured outputs. Tribuo's sequence example support is great
for structured prediction on text, but doesn't integrate well with the rest of the type system.
We'd like to refactor it to make the structure flexible and user controlled, and then extend
out support for ranking tasks and other multi-example tasks.
- Online learning. Tribuo doesn't currently support online learning, and it could be simply
extended to support it when the feature and output domains do not change size (i.e. observing 
new data drawn i.i.d. from the same training distribution). We'd also like to support
online learning in environments with concept drift and dataset shift, but those problems are
harder and so further down the roadmap.
- Add hypothesis testing support to the evaluations.
- Add alternate indexes to `Dataset`. Currently `Dataset` is indexed by an integer id, it's position
in the underlying list. It would be useful to have a Map view of a dataset to subsample it for
specific operations (though this can be achieved today using `DatasetView` and predicates).
- Make `Example`s immutable after they've been added to a `Dataset`. This is likely to be a breaking change.
- Add support for global feature transformations, like normalizing to a unit vector, applying PCA and others.
- Integrate with a plotting library.

## Internals

- Better support different feature types. Currently the feature domain only knows of
categorical and real valued features, and promotes the former to the latter when there
are too many categories. This could be tied into the `RowProcessor` to give the user control
over the feature types, which could filter down into algorithmic choices elsewhere in the package.
- Serialization. We'd like to have alternate serialization mechanisms for models and datasets until
Java's serialization mechanisms improve.
- Caching datasource. Datasources may currently perform expensive feature extraction steps 
(I'm looking at you `RowProcessor`), and it would be useful to be able to cache the output of
that locally, while maintaining the link to the original data. We don't have a firm design for
this feature yet, but we're in need of it for some internal work.
- KMeans & Nearest Neighbour share very little code, but are conceptually very similar. We'd like
to refactor out the shared code (while maintaining serialization compatibility).
- Allow `DatasetView` to regenerate it's feature and output domains. Currently all views of a dataset
share the same immutable feature domain, but in some cases this can leak information from test time
to train (e.g. when using the unselected data as an out of bag sample).
- Fix batch prediction methods so they don't throw `IllegalArgumentException` in the middle of a batch,
and instead return all the valid predictions and a list of the invalid predictions (i.e. ones with invalid 
examples, or examples which didn't have suitable features for the model).

## New ML algorithms or parameters

- Add K-Means++ initialisation for K-Means.
- Add extra parameters to the tree trainers to allow for an ExtraTrees style ensemble, and to 
specify a minimum purity decrease requirement.
- Gaussian Processes.
- Vowpal Wabbit interface.
- Feature selection. We already have several feature selection algorithms implemented 
in a Tribuo compatible interface, but the codebase isn't quite ready for release.
- Support word embedding features.
- Support contextualised word embeddings (through the ONNX or TensorFlow interfaces).
- More complex Multi-Label prediction algorithms.
- More anomaly detection algorithms.
- More clustering algorithms.

## Performance

- Trees:
    - Support parallel training of trees and forests (this is mostly supported at the tree level, 
    but needs some work at the ensemble level).
    - Reduce repeated work in the regression impurity metrics.
    - Prevent wasted computation when computing leaves (leaf nodes have their statistics computed 
 as if they were going to be split, even when it is known they won't due to their size).
- Multithreading the various SGD based trainers using a Hogwild approach.
- Incorporate support for a BLAS.
- Investigate use of the Java Vector API to improve performance critical math operations.

## Documentation

- Fill out the javadoc so it exists for all public and protected methods, including constructors.
- Add more tutorials.

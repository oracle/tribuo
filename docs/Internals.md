# Developing inside Tribuo

Tribuo internally maintains several invariants, and there are a few tricky
parts to writing new `Trainer` and `Model` implementations. This document
covers the internal lifecycle of a training and evaluation run, along with
various considerations when extending Tribuo's classes.

## Implementation Considerations

Tribuo's view of the world is a large, sparse feature space, which is mapped to
a smaller dense output space. It builds the metadata when an `Example` is added
to a `Dataset`, and that metadata is used to inform the training procedure,
along with providing the basis for explanation systems like LIME.

The main invariant is that features are stored in `Example`s in
*lexicographically sorted order* (i.e., using String's natural ordering). This
filters down into the internal feature maps, where the id numbers are assigned
using the same lexicographic ordering (i.e., `AA` = 1, `AB` = 2, ... ,`BA` = 27
etc). Output dimensions are *not ordered*, they usually have ids assigned based
on presentation order in the dataset.  However output dimension ids are less
visible to developers using Tribuo. The feature order invariant is maintained
by all the `Example` subclasses. It's also maintained by the feature name
obfuscation, so the hashed names are assigned ids based on the lexicographic
ordering of the unhashed names. This ensures that any observed feature must
have an id number less than or equal to the previous feature, even after
hashing, which makes operations in `SparseVector` much simpler.

The `Output` subclasses are defined so that `Output.equals` and `Output.hashCode`
only care about the Strings stored in each `Output`. This is so that a 
`Label(name="POSITIVE",value=0.6)` is considered equal to a 
`Label(name="POSITIVE",value=1.0)`, and so the `OutputInfo` which stores
the `Output`s in a hashmap has a consistent view of the world. Comparing
two outputs for total equality (i.e., including any values) should be done
using `Output.fullEquals()`. This approach works well for classification,
anomaly detection and clustering, but for regression tasks, any `Regressor`
is considered equal to any other `Regressor` if they share the same
output dimension names which is rather confusing. A refactor which changed
this behaviour on the `Regressor` would lead to unfortunate interactions with
the other `Output` subclasses, and would involve a backwards incompatible change
to the `OutputInfo` implementations stored inside every `Model`. We plan to 
fix this behaviour when we've found an alternate design which ensures consistency,
but until then this pattern should be followed for any new
`Output` implementations. 

It's best practice not to modify an `Example` after it's been passed to a
`Dataset` except by methods on that dataset. This allows the `Dataset` to track
the feature values, and ensure the metadata is up to date. It's especially
dangerous to add new features to an `Example` inside a `Dataset`, as the
`Dataset` won't have a mapping for the new features, and they are likely to be
ignored. We are likely to enforce this restriction in a future version.

When subclassing one of Tribuo's interfaces or abstract classes, it's important
to implement the provenance object. If the state of the class can be purely
described by configuration, then you can use `ConfiguredObjectProvenanceImpl`
which uses reflection to collect the necessary information. If the state is
partially described by the configuration, then it's trickier, and it's
recommended to subclass `SkeletalConfiguredObjectProvenance` which provides the
reflective parts of the provenance. All provenance classes must have a public
constructor which accepts a `Map<String,Provenance>`, this is used for
serialisation. The other required methods are constrained by the interface.
Provenance classes **must** be immutable after construction.  Provenance is a
record of what has happened to produce a class, and it must not change. The aim
of the provenance system is that it is completely transparent to the users of
the library, it's pervasive and always correct. The user shouldn't have to know
anything about provenance, configuration or tracking of state to have
provenance built into their models and evaluations.

## Tracing a training and evaluation run

This section describes the internal process of a training and evaluation run.

### DataSource 
`Example`s are created in a `DataSource`. Preferably they are created with a
`Feature` list as this ensures the O(n log n) sort cost is paid once, rather than
multiple insertions and sorts. The `DataSource` uses the supplied
`OutputFactory` implementation to convert whatever is denoted as the output
into an `Output` subclass. The `Example` creation process can be loading
tabular data from disk, loading and tokenizing text, parsing a JSON file, etc.
The two requirements are that there is some `DataProvenance` generated and that
`Example`s are produced. The `DataSource` is then fed into a `MutableDataset`.

### Dataset
The `MutableDataset` performs three operations on the `DataSource`: it copies
out the `DataSource`'s `OutputFactory`, it stores the `DataProvenance` of the
source, and it iterates the `DataSource` once loading the `Example`s into the
`Dataset`. First a new `MutableFeatureMap` is created, then the `OutputFactory`
is used to generate a `MutableOutputInfo` of the appropriate type (e.g.
`MutableLabelInfo` for multi-class classification).  Each `Example` has its
`Output` recorded in the `OutputInfo` including checking to see if this
`Example` is unlabelled, denoted by the appropriate "unknown `Output`" sentinel
that each `OutputFactory` implementation can create. Then the `Example`'s
`Feature`s are iterated. Each `Feature` is passed into the `MutableFeatureMap`
where its value and name are recorded. If the `Feature` hasn't been observed
before, then a new `VariableInfo` is created, typically a `CategoricalInfo`,
and the value is recorded. With the default feature map implementation, if a
categorical variable has more than 50 unique values the `CategoricalInfo` is
replaced with a `RealInfo` which treats that feature as real valued. We expect
to provide more control over this transformation in a future release. The
`CategoricalInfo` captures a histogram of the feature values, and the
`RealInfo` tracks max, min, mean and variance. Both track the number of times
this `Feature` was observed.

At this point the `Dataset` can be transformed, by a `TransformationMap`. This
applies an independent sequence of transformation to each `Feature`, so it can
perform rescaling or binning, but not Principal Component Analysis (PCA).  The
`TransformationMap` gathers the necessary statistics about the features, and
then rewrites each `Example` according to the transformation, generating a
`TransformerMap` which can be used to apply that specific transformations to
other `Dataset`s (e.g., to fit the transformation on the training data, and
apply it to the test data), and recording the transformations in the
`Dataset`'s `DataProvenance`. This can also be done at training time using the
`TransformTrainer` which wraps the trained `Model` so the transformations are
always applied.  Transformations which depend on all features and can change
the feature space itself (e.g., PCA or feature selection) are planned for a
future release. 

### Training
On entry into a train method, several things happen: the train invocation count
is incremented, an RNG specific to this method call is split from the
`Trainer`'s RNG if required, the `Dataset` is queried for its
`DataProvenance`, the `Dataset`'s `FeatureMap` is converted into an
`ImmutableFeatureMap`, and the `Dataset`s' `OutputInfo` is converted into an
`ImmutableOutputInfo`. These last two steps involve freezing the feature and
output domains, and assigning ids to each feature and output dimension. Finally
the `OutputInfo` is checked to see if it contains any sentinel unknown
`Output`s. If it does, and the `Trainer` is fully supervised, an exception is
thrown.

The majority of `Trainer`s then create a `SparseVector` from each `Example`'s
features, and copies out the `Output` into either an id or double value
(depending on its class). The `SparseVector` guarantees that there are no id
collisions by adding together colliding feature values (collisions can be
induced by feature hashing), and otherwise validates the `Example`. Ensemble
`Trainer`s and others which wrap an inner `Trainer` leave the `SparseVector`
conversion to the inner `Trainer`.

The `Trainer` then executes its training algorithm to produce the model
parameters.

Finally, the `Trainer` constructs the `ModelProvenance` incorporating the
`Dataset`'s `DataProvenance`, the `Trainer`'s `TrainerProvenance` (i.e.,
training hyperparameters), and any run specific `Provenance` that the user
provided. The `ModelProvenance` along with the `ImmutableFeatureMap`,
`ImmutableOutputInfo`, and the model parameters are supplied to the appropriate
model constructor, and the trained `Model` is returned.

`Model`s are immutable, apart from parameters which control test time behaviour
such as inference batch size, number of inference threads, choice of threading
backend etc.

### Evaluation
Once an `Evaluator` of the appropriate type has been constructed (either
directly or via an `OutputFactory`), a `Model` can be evaluated by an
`Evaluator` on either a `Dataset` or a `DataSource`, the process is similar
either way. The `Model` has its `predict(Iterable<Example>)` method called.
This method first converts each `Example` into a `SparseVector` using the
`Model`'s `ImmutableFeatureMap`. This implicitly checks to see if the `Model`
and the `Example` have any feature overlap, if the `SparseVector` has no active
elements then there is no feature overlap and an exception is thrown.  The
`Model` then produces the prediction using its parameters, and then a
`Prediction` object is created which maps the predicted values to their
dimensions or labels. The `Evaluator` aggregates all the `Prediction`s, checks
if the `Example`s have ground truth labels (if not it throws an exception), and
then calculates the appropriate evaluation statistics (e.g., accuracy & F1 for
classification, RMSE for regression etc). Finally, the input data's
`DataProvenance` and the `Model`'s `ModelProvenance` are queried, and the
evaluation statistics, provenances and predictions are passed to the
appropriate `Evaluation`'s constructor for storage.

## Protobuf Serialization

Tribuo's protobuf serialization is based around redirection and the `Any` packed
protobuf to simulate polymorphic behaviour. Each type is packaged into a top
level protobuf representing the interface it implements which has an integer 
version field incrementing from 0, the class name of the class which can 
deserialize this object, and a packed `Any` message which contains class specific serialization information. This protobuf is
unpacked using the deserialization mechanism in `org.tribuo.protos.ProtoUtil` and
then the method `deserializeFromProto(int version, String className, Any message)`
is called on the `className` specified in the proto. The class name is passed through
to allow redirection for Tribuo internal classes which may want to deserialize as a
different type as we evolve the library. That method then typically checks that the
version is supported by the current class, to prevent inaccurate deserialization of
protobufs written by newer versions of Tribuo when loaded into older versions, and
then the `Any` message is unpacked into a class specific protobuf, any necessary
validation is performed, the deserialized object is constructed and then returned.

There are two helper classes, `ModelDataCarrier` and `DatasetDataCarrier` which
allow easy serialization/deserialization of shared fields in `Model` and 
`Dataset` respectively (and the sequence variants thereof). These are considered
an implementation detail as they may change to incorporate new fields, and may
be converted into records when Tribuo moves to a newer version of Java.

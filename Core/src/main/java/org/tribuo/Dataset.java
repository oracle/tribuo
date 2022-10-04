/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.impl.DatasetDataCarrier;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.DatasetProto;
import org.tribuo.protos.core.ExampleProto;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.transform.TransformStatistics;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationMap;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.TransformerMap;
import org.tribuo.util.Util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * A class for sets of data, which are used to train and evaluate classifiers.
 * <p>
 * Subclass {@link MutableDataset} rather than this class.
 * @param <T> the type of the features in the data set.
 */
public abstract class Dataset<T extends Output<T>> implements Iterable<Example<T>>, ProtoSerializable<DatasetProto>,
    Provenancable<DatasetProvenance>, Serializable {
    private static final long serialVersionUID = 2L;

    private static final Logger logger = Logger.getLogger(Dataset.class.getName());

    /**
     * Users of this RNG should synchronize on the Dataset to prevent replicability issues.
     */
    private static final SplittableRandom rng = new SplittableRandom(Trainer.DEFAULT_SEED);

    /**
     * The data in this data set.
     */
    protected final List<Example<T>> data = new ArrayList<>();

    /**
     * The provenance of the data source, extracted on construction.
     */
    protected final DataProvenance sourceProvenance;

    /**
     * A factory for making {@link OutputInfo} and {@link Output} of the appropriate type.
     */
    protected final OutputFactory<T> outputFactory;

    /**
     * The Tribuo version which originally created this dataset
     */
    protected final String tribuoVersion;

    /**
     * The indices of the shuffled order.
     */
    protected int[] indices = null;

    /**
     * Creates a dataset.
     * @param provenance A description of the data, including preprocessing steps.
     * @param outputFactory The output factory.
     */
    protected Dataset(DataProvenance provenance, OutputFactory<T> outputFactory) {
        this(provenance, outputFactory, Tribuo.VERSION);
    }

    /**
     * Creates a dataset.
     * @param provenance A description of the data, including preprocessing steps.
     * @param outputFactory The output factory.
     * @param tribuoVersion The Tribuo version.
     */
    protected Dataset(DataProvenance provenance, OutputFactory<T> outputFactory, String tribuoVersion) {
        this.sourceProvenance = provenance;
        this.outputFactory = outputFactory;
        this.tribuoVersion = tribuoVersion;
    }

    /**
     * Creates a dataset.
     * @param dataSource the DataSource to use.
     */
    protected Dataset(DataSource<T> dataSource) {
        this(dataSource.getProvenance(),dataSource.getOutputFactory());
    }

    /**
     * Deserializes a dataset proto into a dataset.
     * @param datasetProto The proto to deserialize.
     * @return The dataset.
     */
    public static Dataset<?> deserialize(DatasetProto datasetProto) {
        return ProtoUtil.deserialize(datasetProto);
    }

    /**
     * Reads an instance of {@link DatasetProto} from the supplied path and deserializes it.
     * @param path The path to read.
     * @return The deserialized dataset.
     * @throws IOException If the path could not be read from, or the parsing failed.
     */
    public static Dataset<?> deserializeFromFile(Path path) throws IOException {
        try (InputStream is = new BufferedInputStream(Files.newInputStream(path))) {
            return deserializeFromStream(is);
        }
    }

    /**
     * Reads an instance of {@link DatasetProto} from the supplied input stream and deserializes it.
     * @param is The input stream to read.
     * @return The deserialized dataset.
     * @throws IOException If the stream could not be read from, or the parsing failed.
     */
    public static Dataset<?> deserializeFromStream(InputStream is) throws IOException {
        DatasetProto proto = DatasetProto.parseFrom(is);
        return deserialize(proto);
    }

    /**
     * Serializes this dataset to a {@link DatasetProto} and writes it to the supplied path.
     * @param path The path to write to.
     * @throws IOException If the path could not be written to.
     */
    public void serializeToFile(Path path) throws IOException {
        try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(path))) {
            serializeToStream(os);
        }
    }

    /**
     * Serializes this dataset to a {@link DatasetProto} and writes it to the supplied output stream.
     * <p>
     * Does not close the stream.
     * @param stream The output stream to write to.
     * @throws IOException If the stream could not be written to.
     */
    public void serializeToStream(OutputStream stream) throws IOException {
        DatasetProto proto = serialize();
        proto.writeTo(stream);
    }

    /**
     * A String description of this dataset.
     * @return The description
     */
    public String getSourceDescription() {
        return "Dataset(source="+ sourceProvenance.toString() +")";
    }

    /**
     * The provenance of the data this Dataset contains.
     * @return The data provenance.
     */
    public DataProvenance getSourceProvenance() {
        return sourceProvenance;
    }

    /**
     * Gets the examples as an unmodifiable list. This list will throw an UnsupportedOperationException if any elements
     * are added to it.
     * <p>
     * In other words, using the following to add additional examples to this dataset with throw an exception:
     *
     * {@code dataset.getData().add(example)}
     *
     * Instead, use {@link MutableDataset#add(Example)}.
     *
     * @return The unmodifiable example list.
     */
    public List<Example<T>> getData() {
        return Collections.unmodifiableList(data);
    }

    /**
     * Gets the output factory this dataset contains.
     * @return The output factory.
     */
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    /**
     * Gets the set of outputs that occur in the examples in this dataset.
     *
     * @return the set of outputs that occur in the examples in this dataset.
     */
    public abstract Set<T> getOutputs();

    /**
     * Gets the example at the supplied index.
     * <p>
     * Throws IllegalArgumentException if the index is invalid or outside the bounds.
     * @param index The index of the example.
     * @return The example.
     */
    public Example<T> getExample(int index) {
        if ((index < 0) || (index >= size())) {
            throw new IllegalArgumentException("Example index " + index + " is out of bounds.");  
        }
        return data.get(index);
    }

    /**
     * Gets the size of the data set.
     *
     * @return the size of the data set.
     */
    public int size() {
        return data.size();
    }

    /**
     * Shuffles the indices, or stops shuffling them.
     * <p>
     * The shuffle only affects the iterator, it does not affect
     * {@link Dataset#getExample}.
     * <p>
     * Multiple calls with the argument true will shuffle the dataset multiple times.
     * The RNG is shared across all Dataset instances, so methods which access it are synchronized.
     * <p>
     * Using this method will prevent the provenance system from tracking the exact state of the dataset,
     * which may be important for trainers which depend on the example order, like those
     * using stochastic gradient descent.
     * @param shuffle If true shuffle the data.
     */
    public synchronized void shuffle(boolean shuffle) {
        if (shuffle) {
            indices = Util.randperm(data.size(), rng);
        } else {
            indices = null;
        }
    }

    /**
     * Returns or generates an {@link ImmutableOutputInfo}.
     * @return An immutable output info.
     */
    public abstract ImmutableOutputInfo<T> getOutputIDInfo();

    /**
     * Returns this dataset's {@link OutputInfo}.
     * @return The output info.
     */
    public abstract OutputInfo<T> getOutputInfo();

    /**
     * Returns or generates an {@link ImmutableFeatureMap}.
     * @return An immutable feature map with id numbers.
     */
    public abstract ImmutableFeatureMap getFeatureIDMap();

    /**
     * Returns this dataset's {@link FeatureMap}.
     * @return The feature map from this dataset.
     */
    public abstract FeatureMap getFeatureMap();

    @Override
    public synchronized Iterator<Example<T>> iterator() {
        if (indices == null) {
            return data.iterator();
        } else {
            return new ShuffleIterator<>(this,indices);
        }
    }

    @Override
    public String toString(){
        return "Dataset(source="+ sourceProvenance +")";
    }

    /**
     * Takes a {@link TransformationMap} and converts it into a {@link TransformerMap} by
     * observing all the values in this dataset.
     * <p>
     * Does not mutate the dataset, if you wish to apply the TransformerMap, use
     * {@link MutableDataset#transform} or {@link TransformerMap#transformDataset}.
     * <p>
     * TransformerMaps operate on feature values which are present, sparse values
     * are ignored and not transformed. If the zeros should be transformed, call
     * {@link MutableDataset#densify} on the datasets before applying a transformer.
     * <p>
     * This method calls {@link #createTransformers(TransformationMap, boolean)} with
     * {@code includeImplicitZeroFeatures} set to false, thus ignoring implicitly zero
     * features when fitting the transformations. This is the default behaviour in
     * Tribuo 4.0, but causes erroneous behaviour in
     * {@link org.tribuo.transform.transformations.IDFTransformation} so should be
     * avoided with that transformation.
     * See {@link org.tribuo.transform} for a more detailed discussion of densify and includeImplicitZeroFeatures.
     * <p>
     * Throws {@link IllegalArgumentException} if the TransformationMap object has
     * regexes which apply to multiple features.
     * @param transformations The transformations to fit.
     * @return A TransformerMap which can apply the transformations to a dataset.
     */
    public TransformerMap createTransformers(TransformationMap transformations) {
        return createTransformers(transformations, false);
    }

    /**
     * Takes a {@link TransformationMap} and converts it into a {@link TransformerMap} by
     * observing all the values in this dataset.
     * <p>
     * Does not mutate the dataset, if you wish to apply the TransformerMap, use
     * {@link MutableDataset#transform} or {@link TransformerMap#transformDataset}.
     * <p>
     * TransformerMaps operate on feature values which are present, sparse values
     * are ignored and not transformed. If the zeros should be transformed, call
     * {@link MutableDataset#densify} on the datasets before applying a transformer.
     * See {@link org.tribuo.transform} for a more detailed discussion of densify and includeImplicitZeroFeatures.
     * <p>
     * Throws {@link IllegalArgumentException} if the TransformationMap object has
     * regexes which apply to multiple features.
     * @param transformations The transformations to fit.
     * @param includeImplicitZeroFeatures Use the implicit zero feature values to construct the transformations.
     * @return A TransformerMap which can apply the transformations to a dataset.
     */
    public TransformerMap createTransformers(TransformationMap transformations, boolean includeImplicitZeroFeatures) {
        ArrayList<String> featureNames = new ArrayList<>(getFeatureMap().keySet());

        // Validate map by checking no regex applies to multiple features.
        logger.fine(String.format("Processing %d feature specific transforms", transformations.getFeatureTransformations().size()));
        Map<String,List<Transformation>> featureTransformations = new HashMap<>();
        for (Map.Entry<String,List<Transformation>> entry : transformations.getFeatureTransformations().entrySet()) {
            // Compile the regex.
            Pattern pattern = Pattern.compile(entry.getKey());
            // Check all the feature names
            for (String name : featureNames) {
                // If the regex matches
                if (pattern.matcher(name).matches()) {
                    List<Transformation> oldTransformations = featureTransformations.put(name,entry.getValue());
                    // See if there is already a transformation list for that name.
                    if (oldTransformations != null) {
                        throw new IllegalArgumentException("Feature name '"
                                + name + "' matches multiple regexes, at least one of which was '"
                                + entry.getKey() + "'.");
                    }
                }
            }
        }

        // Populate the feature transforms map.
        Map<String,Queue<TransformStatistics>> featureStats = new HashMap<>();
        // sparseCount tracks how many times a feature was not observed
        Map<String,MutableLong> sparseCount = new HashMap<>();
        for (Map.Entry<String,List<Transformation>> entry : featureTransformations.entrySet()) {
            // Create the queue of feature transformations for this feature
            Queue<TransformStatistics> l = new LinkedList<>();
            for (Transformation t : entry.getValue()) {
                l.add(t.createStats());
            }
            // Add the queue to the map for that feature
            featureStats.put(entry.getKey(),l);
            sparseCount.put(entry.getKey(), new MutableLong(data.size()));
        }
        if (!transformations.getGlobalTransformations().isEmpty()) {
            // Append all the global transformations
            int ntransform = featureNames.size();
            logger.fine(String.format("Starting %,d global transformations", ntransform));
            int ndone = 0;
            for (String v : featureNames) {
                // Create the queue of feature transformations for this feature
                Queue<TransformStatistics> l = featureStats.computeIfAbsent(v, (k) -> new LinkedList<>());
                for (Transformation t : transformations.getGlobalTransformations()) {
                    l.add(t.createStats());
                }
                // Add the queue to the map for that feature
                featureStats.put(v, l);
                // Generate the sparse count initialised to the number of features.
                sparseCount.putIfAbsent(v, new MutableLong(data.size()));
                ndone++;
                if(logger.isLoggable(Level.FINE) && ndone % 10000 == 0) {
                    logger.fine(String.format("Completed %,d of %,d global transformations", ndone, ntransform));
                }
            }
        }

        Map<String,List<Transformer>> output = new LinkedHashMap<>();
        Set<String> removeSet = new LinkedHashSet<>();
        boolean initialisedSparseCounts = false;
        // Iterate through the dataset max(transformations.length) times.
        while (!featureStats.isEmpty()) {
            for (Example<T> example : data) {
                for (Feature f : example) {
                    if (featureStats.containsKey(f.getName())) {
                        if (!initialisedSparseCounts) {
                            sparseCount.get(f.getName()).decrement();
                        }
                        List<Transformer> curTransformers = output.get(f.getName());
                        // Apply all current transformations
                        double fValue = TransformerMap.applyTransformerList(f.getValue(), curTransformers);
                        // Observe the transformed value
                        featureStats.get(f.getName()).peek().observeValue(fValue);
                    }
                }
            }
            // Sparse counts are updated (this could be protected by an if statement)
            initialisedSparseCounts = true;

            removeSet.clear();
            // Emit the new transformers onto the end of the list in the output map.
            for (Map.Entry<String,Queue<TransformStatistics>> entry : featureStats.entrySet()) {
                TransformStatistics currentStats = entry.getValue().poll();
                if (includeImplicitZeroFeatures) {
                    // Observe all the sparse feature values
                    int unobservedFeatures = sparseCount.get(entry.getKey()).intValue();
                    currentStats.observeSparse(unobservedFeatures);
                }
                // Get the transformer list for that feature (if absent)
                List<Transformer> l = output.computeIfAbsent(entry.getKey(), (k) -> new ArrayList<>());
                // Generate the transformer and add it to the appropriate list.
                l.add(currentStats.generateTransformer());
                // If the queue is empty, remove that feature, ensuring that featureStats is eventually empty.
                if (entry.getValue().isEmpty()) {
                    removeSet.add(entry.getKey());
                }
            }
            // Remove the features with empty queues.
            for (String s : removeSet) {
                featureStats.remove(s);
            }
        }

        return new TransformerMap(output,getProvenance(),transformations.getProvenance());
    }

    /**
     * Constructs the data carrier for serialization.
     * @param featureMap The feature domain.
     * @param outputInfo The output domain.
     * @return The serialization data carrier.
     */
    protected DatasetDataCarrier<T> createDataCarrier(FeatureMap featureMap, OutputInfo<T> outputInfo) {
        return createDataCarrier(featureMap,outputInfo,Collections.emptyList());
    }

    /**
     * Constructs the data carrier for serialization.
     * @param featureMap The feature domain.
     * @param outputInfo The output domain.
     * @param transformationProvenances The transformation provenances, must be non-null, but can be empty.
     * @return The serialization data carrier.
     */
    protected DatasetDataCarrier<T> createDataCarrier(FeatureMap featureMap, OutputInfo<T> outputInfo, List<ObjectProvenance> transformationProvenances) {
        String version = tribuoVersion == null ? Tribuo.VERSION : tribuoVersion;
        return new DatasetDataCarrier<>(sourceProvenance,featureMap,outputInfo,outputFactory,transformationProvenances,version);
    }

    /**
     * Validates that this Dataset does in fact contain the supplied output type.
     * <p>
     * As the output type is erased at runtime, deserialising a Dataset is an unchecked
     * operation. This method allows the user to check that the deserialised dataset is
     * of the appropriate type, rather than seeing if the Dataset throws a {@link ClassCastException}
     * when used.
     * @param clazz The class object to verify the output type against.
     * @return True if the output type is assignable to the class object type, false otherwise.
     */
    public boolean validate(Class<? extends Output<?>> clazz) {
        Set<T> domain = getOutputInfo().getDomain();
        boolean output = true;
        for (T type : domain) {
            output &= clazz.isInstance(type);
        }
        return output;
    }

    /**
     * Casts the dataset to the specified output type, assuming it is valid.
     * <p>
     * If it's not valid, throws {@link ClassCastException}.
     * @param inputDataset The model to cast.
     * @param outputType The output type to cast to.
     * @param <T> The output type.
     * @return The model cast to the correct value.
     */
    public static <T extends Output<T>> Dataset<T> castDataset(Dataset<?> inputDataset, Class<T> outputType) {
        if (inputDataset.validate(outputType)) {
            @SuppressWarnings("unchecked") // guarded by validate
            Dataset<T> castedModel = (Dataset<T>) inputDataset;
            return castedModel;
        } else {
            throw new ClassCastException("Attempted to cast dataset to " + outputType.getName() + " which is not valid for dataset " + inputDataset.toString());
        }
    }

    private static class ShuffleIterator<T extends Output<T>> implements Iterator<Example<T>> {
        private final Dataset<T> data;
        private final int[] indices;
        private int index;

        public ShuffleIterator(Dataset<T> data, int[] indices) {
            this.data = data;
            this.indices = indices;
            this.index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < indices.length;
        }

        @Override
        public Example<T> next() {
            Example<T> e = data.getExample(indices[index]);
            index++;
            return e;
        }
    }

    /**
     * Deserializes a list of example protos into a list of examples.
     * @param examplesList The protos.
     * @param outputClass The output class.
     * @param fmap The feature domain.
     * @return The list of deserialized examples.
     */
    protected static List<Example<?>> deserializeExamples(List<ExampleProto> examplesList, Class<?> outputClass, FeatureMap fmap) {
        List<Example<?>> examples = new ArrayList<>();
        for (ExampleProto e : examplesList) {
            Example<?> example = Example.deserialize(e);
            if (example.getOutput().getClass().equals(outputClass)) {
                for (Feature f : example) {
                   if (fmap.get(f.getName()) == null) {
                       throw new IllegalStateException("Invalid protobuf, feature domain does not contain feature " + f.getName() + " present in an example");
                   }
                }
                examples.add(example);
            } else {
                throw new IllegalStateException("Invalid protobuf, expected all examples to have output class " + outputClass + ", but found " + example.getOutput().getClass());
            }
        }
        return examples;
    }
}


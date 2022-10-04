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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.hash.Hasher;
import org.tribuo.impl.DatasetDataCarrier;
import org.tribuo.protos.core.DatasetProto;
import org.tribuo.protos.core.ImmutableDatasetProto;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.util.Merger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * This is a {@link Dataset} which has an {@link ImmutableFeatureMap} to store the feature information.
 * Whenever an example is added to this dataset it removes features that do not exist in the {@link FeatureMap}.
 * The dataset is immutable after construction (unless the examples are modified).
 * <p>
 * This class is mostly for performance optimisations inside the framework, and should not
 * generally be used by external code.
 */
public class ImmutableDataset<T extends Output<T>> extends Dataset<T> implements Serializable {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(ImmutableDataset.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Output information, and id numbers for outputs found in this dataset.
     */
    protected ImmutableOutputInfo<T> outputIDInfo;

    /**
     * A map from feature names to IDs for the features found in this dataset.
     */
    protected ImmutableFeatureMap featureIDMap;

    /**
     * If true, instead of throwing an exception when an invalid {@link Example} is encountered, this Dataset will log a warning and drop it.
     */
    protected final boolean dropInvalidExamples;

    private DatasetProvenance provenance = null;

    /**
     * If you call this it's your job to setup outputMap, featureIDMap and fill it with examples.
     * <p>
     * Note: Sets dropInvalidExamples to false.
     *
     * @param description A description of the input data (including preprocessing steps).
     * @param outputFactory The factory for this output type.
     */
    protected ImmutableDataset(DataProvenance description, OutputFactory<T> outputFactory) {
        super(description,outputFactory);
        dropInvalidExamples = false;
    }

    /**
     * Creates a dataset from a data source. It copies the feature and output maps
     * from the supplied model.
     * @param dataSource The examples.
     * @param model A model to extract feature and output maps from.
     * @param dropInvalidExamples If true, instead of throwing an exception when an invalid {@link Example} is encountered, this Dataset will log a warning and drop it.
     */
    public ImmutableDataset(DataSource<T> dataSource, Model<T> model, boolean dropInvalidExamples) {
        this(dataSource,dataSource.getProvenance(),dataSource.getOutputFactory(),model.getFeatureIDMap(),model.getOutputIDInfo(),dropInvalidExamples);
    }

    /**
     * Creates a dataset from a data source. Creates immutable feature and output maps from the
     * supplied ones.
     * @param dataSource The examples.
     * @param featureIDMap The feature map.
     * @param outputIDInfo The output map.
     * @param dropInvalidExamples If true, instead of throwing an exception when an invalid {@link Example} is encountered, this Dataset will log a warning and drop it.
     */
    public ImmutableDataset(DataSource<T> dataSource, FeatureMap featureIDMap, OutputInfo<T> outputIDInfo, boolean dropInvalidExamples) {
        this(dataSource,dataSource.getProvenance(),dataSource.getOutputFactory(),featureIDMap,outputIDInfo, dropInvalidExamples);
    }

    /**
     * Creates a dataset from a data source. Creates immutable feature and output maps from the
     * supplied ones.
     * @param dataSource The examples.
     * @param description A description of the input data (including preprocessing steps).
     * @param outputFactory The output factory.
     * @param featureIDMap The feature id map, used to remove unknown features.
     * @param outputIDInfo The output id map.
     * @param dropInvalidExamples If true, instead of throwing an exception when an invalid {@link Example} is encountered, this Dataset will log a warning and drop it.
     */
    public ImmutableDataset(Iterable<Example<T>> dataSource, DataProvenance description, OutputFactory<T> outputFactory, FeatureMap featureIDMap, OutputInfo<T> outputIDInfo, boolean dropInvalidExamples) {
        this(dataSource,description, outputFactory, new ImmutableFeatureMap(featureIDMap), outputIDInfo.generateImmutableOutputInfo(), dropInvalidExamples);
    }

    /**
     * Creates a dataset from a data source.
     * @param dataSource The examples.
     * @param description A description of the input data (including preprocessing steps).
     * @param outputFactory The factory for this output type.
     * @param featureIDMap The feature id map, used to remove unknown features.
     * @param outputIDInfo The output id map.
     * @param dropInvalidExamples If true, instead of throwing an exception when an invalid {@link Example} is encountered, this Dataset will log a warning and drop it.
     */
    public ImmutableDataset(Iterable<Example<T>> dataSource, DataProvenance description, OutputFactory<T> outputFactory, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, boolean dropInvalidExamples) {
        super(description,outputFactory);
        this.featureIDMap = featureIDMap;
        this.outputIDInfo = outputIDInfo;
        this.dropInvalidExamples = dropInvalidExamples;

        for (Example<T> ex : dataSource) {
            add(ex);
        }
    }

    /**
     * This is dangerous, and should not be used unless you've overridden everything in ImmutableDataset.
     * <p>
     * Note: Sets dropInvalidExamples to false.
     *
     * @param description A description of the data you're going to add to this dataset.
     * @param outputFactory The factory for this output type.
     * @param featureIDMap The feature id map, used to remove unknown features.
     * @param outputIDInfo The output id map.
     */
    protected ImmutableDataset(DataProvenance description, OutputFactory<T> outputFactory, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo) {
        super(description, outputFactory);
        this.featureIDMap = featureIDMap;
        this.outputIDInfo = outputIDInfo;
        this.dropInvalidExamples = false;
    }

    /**
     * Deserialization constructor.
     * @param provenance The source provenance.
     * @param factory The output factory.
     * @param tribuoVersion The tribuo version.
     * @param fmap The feature id map.
     * @param outputInfo The output id info.
     * @param examples The examples.
     * @param dropInvalidExamples Should invalid examples be dropped when added?
     */
    protected ImmutableDataset(DataProvenance provenance, OutputFactory<T> factory, String tribuoVersion, ImmutableFeatureMap fmap, ImmutableOutputInfo<T> outputInfo, List<Example<T>> examples, boolean dropInvalidExamples) {
        super(provenance,factory,tribuoVersion);
        this.featureIDMap = fmap;
        this.outputIDInfo = outputInfo;
        this.data.addAll(examples);
        this.dropInvalidExamples = dropInvalidExamples;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"unchecked","rawtypes"}) // guarded & checked by getClass checks.
    public static ImmutableDataset<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ImmutableDatasetProto proto = message.unpack(ImmutableDatasetProto.class);
        DatasetDataCarrier<?> carrier = DatasetDataCarrier.deserialize(proto.getMetadata());
        Class<?> outputClass = carrier.outputFactory().getUnknownOutput().getClass();
        FeatureMap fmap = carrier.featureDomain();
        List<Example<?>> examples = deserializeExamples(proto.getExamplesList(), outputClass, fmap);
        if (!(fmap instanceof ImmutableFeatureMap)) {
            throw new IllegalStateException("Invalid protobuf, feature map was not immutable");
        }
        if (!(carrier.outputDomain() instanceof ImmutableOutputInfo)) {
            throw new IllegalStateException("Invalid protobuf, output info was not immutable");
        }
        return new ImmutableDataset(carrier.provenance(), carrier.outputFactory(), carrier.tribuoVersion(), (ImmutableFeatureMap) fmap, (ImmutableOutputInfo) carrier.outputDomain(), examples, proto.getDropInvalidExamples());
    }

    /**
     * Adds an {@link Example} to the dataset, which will remove features with
     * unknown names.
     * @param ex An {@link Example} to add to the dataset.
     */
    protected void add(Example<T> ex) {
        if (!ex.validateExample()) {
            if (dropInvalidExamples) {
                logger.severe("Dropping invalid Example: " + ex.toString());
                return;
            } else {
                throw new IllegalArgumentException("Example had duplicate features, invalid features or no features.");
            }
        }
        innerAdd(ex);
    }

    /**
     * Adds an {@link Example} to the dataset. Use only
     * when the example has already been validated.
     * @param ex An {@link Example} to add to the dataset.
     */
    private void unsafeAdd(Example<T> ex) {
        data.add(ex);
    }

    /**
     * Adds a {@link Example} to the dataset, which will insert feature ids,
     * remove unknown features and sort the examples by the feature ids (merging duplicate ids).
     * @param ex The example to add.
     * @param merger The {@link Merger} to use.
     */
    protected void add(Example<T> ex, Merger merger) {
        ex.reduceByName(merger);
        innerAdd(ex);
    }

    private void innerAdd(Example<T> ex) {
        //
        // Find and remove features that aren't in the feature domain of this dataset.
        List<Feature> featuresToRemove = new ArrayList<>();
        for (Feature f : ex) {
            VariableInfo info = featureIDMap.get(f.getName());
            if (info == null) {
                featuresToRemove.add(f);
            }
        }
        ex.removeFeatures(featuresToRemove);
        //
        // Handle case where Example is empty after removing out-of-domain features.
        if (ex.size() == 0) {
            if (dropInvalidExamples) {
                logger.severe("Dropping invalid Example: " + ex.toString() + ", invalid features - " + featuresToRemove);
            } else {
                throw new IllegalArgumentException("This Dataset does not know any of the Features in this Example.");
            }
        } else {
            ex.canonicalize(featureIDMap);
            data.add(ex);
        }
    }

    @Override
    public Set<T> getOutputs() {
        return outputIDInfo.getDomain();
    }

    @Override
    public ImmutableFeatureMap getFeatureIDMap() {
        return featureIDMap;
    }

    @Override
    public ImmutableFeatureMap getFeatureMap() {
        return featureIDMap;
    }

    @Override
    public ImmutableOutputInfo<T> getOutputIDInfo() {
        return outputIDInfo;
    }

    @Override
    public ImmutableOutputInfo<T> getOutputInfo() {
        return outputIDInfo;
    }

    /**
     * Returns true if this immutable dataset dropped any invalid examples on construction.
     * @return True if it drops invalid examples.
     */
    public boolean getDropInvalidExamples() {
        return dropInvalidExamples;
    }

    @Override
    public String toString() {
        return String.format("ImmutableDataset(source=%s,dropInvalidExamples=%b)", sourceProvenance, dropInvalidExamples);
    }

    @Override
    public synchronized DatasetProvenance getProvenance() {
        if (provenance == null) {
            provenance = cacheProvenance();
        }
        return provenance;
    }

    /**
     * Computes the DatasetProvenance.
     * @return A new dataset provenance.
     */
    private DatasetProvenance cacheProvenance() {
        return new DatasetProvenance(sourceProvenance,new ListProvenance<>(),this);
    }

    @Override
    public DatasetProto serialize() {
        ImmutableDatasetProto.Builder datasetBuilder = ImmutableDatasetProto.newBuilder();

        datasetBuilder.setDropInvalidExamples(dropInvalidExamples);
        datasetBuilder.setMetadata(createDataCarrier(featureIDMap,outputIDInfo).serialize());
        for (Example<T> e : data) {
            datasetBuilder.addExamples(e.serialize());
        }

        DatasetProto.Builder builder = DatasetProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(ImmutableDataset.class.getName());
        builder.setSerializedData(Any.pack(datasetBuilder.build()));

        return builder.build();
    }

    /**
     * Creates an immutable deep copy of the supplied dataset.
     * @param dataset The dataset to copy.
     * @param <T> The type of output.
     * @return An immutable copy of the dataset.
     */
    public static <T extends Output<T>> ImmutableDataset<T> copyDataset(Dataset<T> dataset) {
        ImmutableDataset<T> copy = new ImmutableDataset<>(dataset.getProvenance(),dataset.outputFactory,dataset.getFeatureIDMap(),dataset.getOutputIDInfo());
        for (Example<T> e : dataset) {
            copy.unsafeAdd(e.copy());
        }
        return copy;
    }

    /**
     * Creates an immutable deep copy of the supplied dataset, using a different feature and output map.
     * @param dataset The dataset to copy.
     * @param featureIDMap The new feature map to use. Removes features which are not found in this map.
     * @param outputIDInfo The new output info to use.
     * @param <T> The type of output.
     * @return An immutable copy of the dataset.
     */
    public static <T extends Output<T>> ImmutableDataset<T> copyDataset(Dataset<T> dataset, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo) {
        ImmutableDataset<T> copy = new ImmutableDataset<>(dataset.getProvenance(),dataset.outputFactory,featureIDMap,outputIDInfo);
        for (Example<T> e : dataset) {
            copy.add(e.copy());
        }
        return copy;
    }

    /**
     * Creates an immutable deep copy of the supplied dataset.
     * @param dataset The dataset to copy.
     * @param featureIDMap The new feature map to use. Removes features which are not found in this map.
     * @param outputIDInfo The new output info to use.
     * @param merger The merge function to use to reduce features given new ids.
     * @param <T> The type of output.
     * @return An immutable copy of the dataset.
     */
    public static <T extends Output<T>> ImmutableDataset<T> copyDataset(Dataset<T> dataset, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, Merger merger) {
        ImmutableDataset<T> copy = new ImmutableDataset<>(dataset.getProvenance(),dataset.outputFactory,featureIDMap,outputIDInfo);
        for (Example<T> e : dataset) {
            copy.add(e.copy(),merger);
        }
        return copy;
    }

    /**
     * Creates an immutable shallow copy of the supplied dataset, using the hasher to generate a
     * {@link HashedFeatureMap} which transparently maps from the feature name to the hashed variant.
     * @param dataset The dataset to copy.
     * @param hasher The hashing function to use.
     * @param <T> The type of output.
     * @return An immutable copy of the dataset.
     */
    public static <T extends Output<T>> ImmutableDataset<T> hashFeatureMap(Dataset<T> dataset, Hasher hasher) {
        ImmutableFeatureMap featureIDMap = HashedFeatureMap.generateHashedFeatureMap(dataset.getFeatureMap(),hasher);
        ImmutableDataset<T> copy = new ImmutableDataset<>(dataset.getProvenance(),dataset.outputFactory,featureIDMap,dataset.getOutputIDInfo());
        for (Example<T> e : dataset) {
            copy.unsafeAdd(e);
        }
        return copy;
    }
}

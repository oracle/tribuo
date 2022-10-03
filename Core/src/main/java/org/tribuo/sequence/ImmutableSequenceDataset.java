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

package org.tribuo.sequence;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.VariableInfo;
import org.tribuo.impl.DatasetDataCarrier;
import org.tribuo.protos.core.ImmutableSequenceDatasetProto;
import org.tribuo.protos.core.SequenceDatasetProto;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.util.Merger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * This is a {@link SequenceDataset} which has an {@link ImmutableFeatureMap} to store the feature information.
 * Whenever an example is added to this dataset it removes features that do not exist in the FeatureMap.
 * The dataset is immutable after construction (unless the examples are modified).
 */
public class ImmutableSequenceDataset<T extends Output<T>> extends SequenceDataset<T> implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * A map from labels to IDs for the labels found in this dataset.
     */
    protected ImmutableOutputInfo<T> outputIDInfo;

    /**
     * A map from feature names to IDs for the features found in this dataset.
     */
    protected ImmutableFeatureMap featureIDMap;

    private DatasetProvenance provenance;

    /**
     * If you call this it's your job to setup outputIDInfo and featureIDMap.
     * @param sourceProvenance A description of the dataset including preprocessing steps.
     * @param outputFactory The output factory.
     */
    protected ImmutableSequenceDataset(DataProvenance sourceProvenance, OutputFactory<T> outputFactory) {
        super(sourceProvenance,outputFactory);
    }

    /**
     * Creates a dataset from a data source, taking the output and feature domains from the supplied model.
     * @param dataSource The input data.
     * @param model The model to use for the feature and output domains.
     */
    public ImmutableSequenceDataset(SequenceDataSource<T> dataSource, SequenceModel<T> model) {
        this(dataSource,dataSource.getProvenance(),model.getFeatureIDMap(),model.getOutputIDInfo(),dataSource.getOutputFactory());
    }

    /**
     * Creates a dataset from a data source, using the specified output and feature domains.
     * @param dataSource The input data.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     */
    public ImmutableSequenceDataset(SequenceDataSource<T> dataSource, FeatureMap featureIDMap, OutputInfo<T> outputIDInfo) {
        this(dataSource,dataSource.getProvenance(),featureIDMap,outputIDInfo,dataSource.getOutputFactory());
    }

    /**
     * Creates a dataset from a data source. This method will create the output
     * and feature ID maps that are needed for training and evaluating classifiers.
     * @param dataSource The input data.
     * @param sourceProvenance A description of the data.
     * @param featureIDMap The feature map, used to remove unknown features.
     * @param outputIDInfo The output map.
     * @param outputFactory The output factory.
     */
    public ImmutableSequenceDataset(Iterable<SequenceExample<T>> dataSource, DataProvenance sourceProvenance, FeatureMap featureIDMap, OutputInfo<T> outputIDInfo, OutputFactory<T> outputFactory) {
        this(dataSource,sourceProvenance, new ImmutableFeatureMap(featureIDMap), outputIDInfo.generateImmutableOutputInfo(),outputFactory);
    }

    /**
     * Creates a dataset from a data source.
     * @param dataSource The input data.
     * @param sourceProvenance A description of the data.
     * @param featureIDMap The feature id map, used to remove unknown features.
     * @param outputIDInfo The output id map.
     * @param outputFactory The output factory.
     */
    public ImmutableSequenceDataset(Iterable<SequenceExample<T>> dataSource, DataProvenance sourceProvenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, OutputFactory<T> outputFactory) {
        super(sourceProvenance,outputFactory);
        this.featureIDMap = featureIDMap;
        this.outputIDInfo = outputIDInfo;

        for (SequenceExample<T> ex : dataSource) {
            add(ex);
        }
    }

    /**
     * This is dangerous, and should not be used unless you've overridden everything in ImmutableSequenceDataset.
     * @param sourceProvenance A description of the data, including all preprocessing.
     * @param featureIDMap The feature id map, used to remove unknown features.
     * @param outputIDInfo The output id map.
     */
    protected ImmutableSequenceDataset(DataProvenance sourceProvenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo) {
        super(sourceProvenance,null);
        this.featureIDMap = featureIDMap;
        this.outputIDInfo = outputIDInfo;
    }

    /**
     * Deserialization constructor.
     * @param provenance The source provenance.
     * @param factory The output factory.
     * @param tribuoVersion The tribuo version.
     * @param fmap The feature id map.
     * @param outputInfo The output id info.
     * @param examples The examples.
     */
    protected ImmutableSequenceDataset(DataProvenance provenance, OutputFactory<T> factory, String tribuoVersion, ImmutableFeatureMap fmap, ImmutableOutputInfo<T> outputInfo, List<SequenceExample<T>> examples) {
        super(provenance,factory,tribuoVersion);
        this.featureIDMap = fmap;
        this.outputIDInfo = outputInfo;
        this.data.addAll(examples);
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
    public static ImmutableSequenceDataset<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ImmutableSequenceDatasetProto proto = message.unpack(ImmutableSequenceDatasetProto.class);
        DatasetDataCarrier<?> carrier = DatasetDataCarrier.deserialize(proto.getMetadata());
        Class<?> outputClass = carrier.outputFactory().getUnknownOutput().getClass();
        FeatureMap fmap = carrier.featureDomain();
        List<SequenceExample<?>> examples = deserializeExamples(proto.getExamplesList(), outputClass, fmap);
        if (!(fmap instanceof ImmutableFeatureMap)) {
            throw new IllegalStateException("Invalid protobuf, feature map was not immutable");
        }
        if (!(carrier.outputDomain() instanceof ImmutableOutputInfo)) {
            throw new IllegalStateException("Invalid protobuf, output info was not immutable");
        }
        return new ImmutableSequenceDataset(carrier.provenance(), carrier.outputFactory(), carrier.tribuoVersion(), (ImmutableFeatureMap) fmap, (ImmutableOutputInfo) carrier.outputDomain(), examples);
    }

    /**
     * Adds a {@link SequenceExample} to the dataset, which will insert feature ids, remove unknown features
     * and sort the examples by the feature ids.
     * @param ex The example to add.
     */
    protected void add(SequenceExample<T> ex) {
        if (ex.size() > 0) {
            List<Feature> featuresToRemove = new ArrayList<>();
            for (Example<T> e : ex) {
                featuresToRemove.clear();
                for (Feature f : e) {
                    VariableInfo info = featureIDMap.get(f.getName());
                    if (info == null) {
                        featuresToRemove.add(f);
                    }
                }
                e.removeFeatures(featuresToRemove);
                if (!e.validateExample()) {
                    throw new IllegalStateException("Duplicate features or invalid features inside the Example, or all features were removed.");
                }
            }
            data.add(ex);
            ex.canonicalise(featureIDMap);
        } else {
            throw new IllegalArgumentException("SequenceExample is empty.");
        }
    }

    /**
     * Adds an {@link SequenceExample} to the dataset. Use only
     * when the example has already been validated.
     * @param ex An {@link SequenceExample} to add to the dataset.
     */
    private void unsafeAdd(SequenceExample<T> ex) {
        data.add(ex);
    }

    /**
     * Adds a {@link SequenceExample} to the dataset, which will insert feature ids, remove unknown features
     * and sort the examples by the feature ids.
     * @param ex The example to add.
     * @param merger The merger to use to remove duplicate features.
     */
    protected void add(SequenceExample<T> ex, Merger merger) {
        if (ex.size() > 0) {
            data.add(ex);
            List<Feature> featuresToRemove = new ArrayList<>();
            for (Example<T> e : ex) {
                featuresToRemove.clear();
                for (Feature f : e) {
                    VariableInfo info = featureIDMap.get(f.getName());
                    if (info == null) {
                        featuresToRemove.add(f);
                    }
                }
                e.removeFeatures(featuresToRemove);
                e.reduceByName(merger);
                if (!e.validateExample()) {
                    throw new IllegalStateException("Duplicate features or invalid features inside the Example, or all features were removed.");
                }
            }
        } else {
            throw new IllegalArgumentException("SequenceExample is empty.");
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

    @Override
    public String toString(){
        return "ImmutableSequenceDataset(source="+ sourceProvenance.toString()+")";
    }

    @Override
    public SequenceDatasetProto serialize() {
        ImmutableSequenceDatasetProto.Builder datasetBuilder = ImmutableSequenceDatasetProto.newBuilder();

        datasetBuilder.setMetadata(createDataCarrier(featureIDMap,outputIDInfo).serialize());
        for (SequenceExample<T> e : data) {
            datasetBuilder.addExamples(e.serialize());
        }

        SequenceDatasetProto.Builder builder = SequenceDatasetProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(ImmutableSequenceDataset.class.getName());
        builder.setSerializedData(Any.pack(datasetBuilder.build()));

        return builder.build();
    }

    @Override
    public synchronized DatasetProvenance getProvenance() {
        if (provenance == null) {
            provenance = cacheProvenance();
        }
        return provenance;
    }

    private DatasetProvenance cacheProvenance() {
        return new DatasetProvenance(sourceProvenance,new ListProvenance<>(),this);
    }

    /**
     * Creates an immutable deep copy of the supplied dataset.
     * @param dataset The dataset to copy.
     * @param <T> The type of output.
     * @return An immutable copy of the dataset.
     */
    public static <T extends Output<T>> ImmutableSequenceDataset<T> copyDataset(SequenceDataset<T> dataset) {
        ArrayList<SequenceExample<T>> newData = new ArrayList<>();
        for (SequenceExample<T> e : dataset) {
            newData.add(e.copy());
        }
        return new ImmutableSequenceDataset<>(newData,dataset.getSourceProvenance(),dataset.getFeatureIDMap(),dataset.getOutputInfo(),dataset.getOutputFactory());
    }

    /**
     * Creates an immutable deep copy of the supplied dataset, using a different feature and output map.
     * @param dataset The dataset to copy.
     * @param featureIDMap The new feature map to use. Removes features which are not found in this map.
     * @param outputIDInfo The new output info to use.
     * @param <T> The type of output.
     * @return An immutable copy of the dataset.
     */
    public static <T extends Output<T>> ImmutableSequenceDataset<T> copyDataset(SequenceDataset<T> dataset, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo) {
        ArrayList<SequenceExample<T>> newData = new ArrayList<>();
        for (SequenceExample<T> e : dataset) {
            newData.add(e.copy());
        }
        return new ImmutableSequenceDataset<>(newData,dataset.getSourceProvenance(),featureIDMap,outputIDInfo,dataset.getOutputFactory());
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
    public static <T extends Output<T>> ImmutableSequenceDataset<T> copyDataset(SequenceDataset<T> dataset, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, Merger merger) {
        ImmutableSequenceDataset<T> copy = new ImmutableSequenceDataset<>(dataset.getProvenance(),featureIDMap,outputIDInfo);
        for (SequenceExample<T> e : dataset) {
            copy.add(e.copy(),merger);
        }
        return copy;
    }

    static <T extends Output<T>> ImmutableSequenceDataset<T> changeFeatureMap(SequenceDataset<T> dataset, ImmutableFeatureMap featureIDMap) {
        ImmutableSequenceDataset<T> copy = new ImmutableSequenceDataset<>(dataset.getProvenance(),featureIDMap,dataset.getOutputIDInfo());
        for (SequenceExample<T> e : dataset) {
            copy.unsafeAdd(e);
        }
        return copy;
    }
}

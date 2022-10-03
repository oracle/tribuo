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

package org.tribuo.dataset;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.VariableInfo;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.DatasetDataCarrier;
import org.tribuo.protos.core.DatasetProto;
import org.tribuo.protos.core.MinimumCardinalityDatasetProto;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;

/**
 * This class creates a pruned dataset in which low frequency features that
 * occur less than the provided minimum cardinality have been removed. This can
 * be useful when the dataset is very large due to many low-frequency features.
 * For example, this class can be used to remove low frequency words from a BoW
 * formatted dataset. Here, a new dataset is created so that the feature counts
 * are recalculated and so that the original, passed-in dataset is not modified.
 * The returned dataset may have fewer examples because if any of the examples
 * have no features after the minimum cardinality has been applied, then those
 * examples will not be added to the constructed dataset.
 * 
 * @param <T> The type of the outputs in this {@link Dataset}.
 */
public class MinimumCardinalityDataset<T extends Output<T>> extends ImmutableDataset<T> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(MinimumCardinalityDataset.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final int minCardinality;

    private int numExamplesRemoved = 0;

    private final Set<String> removed = new HashSet<>();

    /**
     * @param dataset this dataset is left untouched and is used to populate the constructed dataset.
     * @param minCardinality features with a frequency less than minCardinality will be removed.  
     */
    public MinimumCardinalityDataset(Dataset<T> dataset, int minCardinality) {
        super(dataset.getProvenance(), dataset.getOutputFactory());
        this.minCardinality = minCardinality;
        if (minCardinality < 0) {
            throw new IllegalArgumentException("Minimum cardinality must be non-negative, found " + minCardinality);
        }

        MutableFeatureMap featureInfos = new MutableFeatureMap();

        List<Feature> features = new ArrayList<>();
        //
        // Rebuild the data list only with features that have a minimum cardinality.
        FeatureMap wfm = dataset.getFeatureMap();
        for (Example<T> ex : dataset) {
            features.clear();
            ArrayExample<T> nex = new ArrayExample<>(ex.getOutput());
            nex.setWeight(ex.getWeight());
            for (Feature f : ex) {
                VariableInfo finfo = wfm.get(f.getName());
                if (finfo == null || finfo.getCount() < minCardinality) {
                    //
                    // The feature info might be null if we have a feature at 
                    // prediction time that we didn't see
                    // at training time.
                    removed.add(f.getName());
                } else {
                    features.add(f);
                }
            }
            nex.addAll(features);
            if (nex.size() > 0) {
                if (!nex.validateExample()) {
                    throw new IllegalStateException("Duplicate features found in example " + nex.toString());
                }
                data.add(nex);
            } else {
                numExamplesRemoved++;
            }
        }

        // Copy out the feature infos above the threshold.
        for (VariableInfo info : wfm) {
            if (info.getCount() >= minCardinality) {
                featureInfos.put(info.copy());
            }
        }

        outputIDInfo = dataset.getOutputIDInfo();
        featureIDMap = new ImmutableFeatureMap(featureInfos);

        if(numExamplesRemoved > 0) {
            logger.info(String.format("filtered out %d examples because it had zero features after the minimum frequency count was applied.", numExamplesRemoved));
        }

    }

    private MinimumCardinalityDataset(DataProvenance provenance, OutputFactory<T> factory, String tribuoVersion,
                                      ImmutableFeatureMap fmap, ImmutableOutputInfo<T> outputInfo,
                                      List<Example<T>> examples, boolean dropInvalidExamples, int minCardinality,
                                      int numExamplesRemoved, Set<String> removed) {
        super(provenance,factory,tribuoVersion,fmap,outputInfo,examples,dropInvalidExamples);
        this.minCardinality = minCardinality;
        this.numExamplesRemoved = numExamplesRemoved;
        this.removed.addAll(removed);
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
    public static MinimumCardinalityDataset<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        MinimumCardinalityDatasetProto proto = message.unpack(MinimumCardinalityDatasetProto.class);
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
        int minCardinality = proto.getMinCardinality();
        if (minCardinality < 0) {
            throw new IllegalStateException("Invalid protobuf, minimum cardinality must be non-negative, found " + minCardinality);
        }
        int numRemoved = proto.getNumExamplesRemoved();
        if (numRemoved < 0) {
            throw new IllegalStateException("Invalid protobuf, number of examples removed must be non-negative, found " + numRemoved);
        }
        List<String> removedList = proto.getRemovedList();
        Set<String> removedFeatures = new HashSet<>(removedList);
        if (removedFeatures.size() != removedList.size()) {
            throw new IllegalStateException("Invalid protobuf, features removed contained duplicates, removed = " + removedList);
        }
        return new MinimumCardinalityDataset(carrier.provenance(), carrier.outputFactory(), carrier.tribuoVersion(),
            (ImmutableFeatureMap) fmap, (ImmutableOutputInfo) carrier.outputDomain(), examples,
            proto.getDropInvalidExamples(), minCardinality, numRemoved, removedFeatures);
    }

    /**
     * The feature names that were removed.
     * @return The feature names.
     */
    public Set<String> getRemoved() {
        return removed;
    }

    /**
     * The number of examples removed due to a lack of features.
     * @return The number of removed examples.
     */
    public int getNumExamplesRemoved() {
        return numExamplesRemoved;
    }

    /**
     * The minimum cardinality threshold for the features.
     * @return The cardinality threshold.
     */
    public int getMinCardinality() {
        return minCardinality;
    }

    @Override
    public DatasetProvenance getProvenance() {
        return new MinimumCardinalityDatasetProvenance(this);
    }

    @Override
    public DatasetProto serialize() {
        MinimumCardinalityDatasetProto.Builder datasetBuilder = MinimumCardinalityDatasetProto.newBuilder();

        datasetBuilder.setDropInvalidExamples(dropInvalidExamples);
        datasetBuilder.setMetadata(createDataCarrier(featureIDMap,outputIDInfo).serialize());
        for (Example<T> e : data) {
            datasetBuilder.addExamples(e.serialize());
        }
        datasetBuilder.setMinCardinality(minCardinality);
        datasetBuilder.setNumExamplesRemoved(numExamplesRemoved);
        datasetBuilder.addAllRemoved(removed);

        DatasetProto.Builder builder = DatasetProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(MinimumCardinalityDataset.class.getName());
        builder.setSerializedData(Any.pack(datasetBuilder.build()));

        return builder.build();
    }

    /**
     * Provenance for {@link MinimumCardinalityDataset}.
     */
    public static class MinimumCardinalityDatasetProvenance extends DatasetProvenance {
        private static final long serialVersionUID = 1L;

        private static final String MIN_CARDINALITY = "min-cardinality";

        private final IntProvenance minCardinality;

        <T extends Output<T>> MinimumCardinalityDatasetProvenance(MinimumCardinalityDataset<T> dataset) {
            super(dataset.sourceProvenance, new ListProvenance<>(), dataset);
            this.minCardinality = new IntProvenance(MIN_CARDINALITY,dataset.minCardinality);
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public MinimumCardinalityDatasetProvenance(Map<String,Provenance> map) {
            super(map);
            this.minCardinality = ObjectProvenance.checkAndExtractProvenance(map,MIN_CARDINALITY,IntProvenance.class,MinimumCardinalityDatasetProvenance.class.getSimpleName());
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof MinimumCardinalityDatasetProvenance)) return false;
            if (!super.equals(o)) return false;
            MinimumCardinalityDatasetProvenance pairs = (MinimumCardinalityDatasetProvenance) o;
            return minCardinality.equals(pairs.minCardinality);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), minCardinality);
        }

        @Override
        protected List<Pair<String, Provenance>> allProvenances() {
            List<Pair<String,Provenance>> provenances = super.allProvenances();
            provenances.add(new Pair<>(MIN_CARDINALITY,minCardinality));
            return provenances;
        }
    }
}

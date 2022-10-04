/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableFeatureMap;
import org.tribuo.MutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.DatasetDataCarrier;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.DatasetProto;
import org.tribuo.protos.core.ExampleProto;
import org.tribuo.protos.core.SelectedFeatureDatasetProto;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.FeatureSetProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;

/**
 * This class creates a pruned dataset which only contains the selected features.
 * The new dataset may have fewer examples because if any of the examples
 * have no features after the minimum cardinality has been applied, then those
 * examples will not be added to the constructed dataset.
 * 
 * @param <T> The type of the outputs in this {@link Dataset}.
 */
public final class SelectedFeatureDataset<T extends Output<T>> extends ImmutableDataset<T> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(SelectedFeatureDataset.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final int k;

    private final SelectedFeatureSet featureSet;

    private final Set<String> selectedFeatures;

    private final int numExamplesRemoved;

    /**
     * Constructs a selected feature dataset using all the features in the supplied feature set.
     * @param dataset The dataset to copy.
     * @param featureSet The feature set to use.
     */
    public SelectedFeatureDataset(Dataset<T> dataset, SelectedFeatureSet featureSet) {
        this(dataset,featureSet,-1);
    }

    /**
     * Constructs a selected feature dataset.
     * @param dataset This dataset is left untouched and is used to populate the constructed dataset.
     * @param featureSet The feature set to use.
     * @param k Use the top k features if the feature set is ordered, or {@link FeatureSelector#SELECT_ALL} to select
     *          all of them, throws {@link IllegalArgumentException} if it is unordered and set to a positive value.
     */
    public SelectedFeatureDataset(Dataset<T> dataset, SelectedFeatureSet featureSet, int k) {
        super(dataset.getProvenance(), dataset.getOutputFactory());
        this.featureSet = featureSet;
        this.k = k;

        // Validate feature set & k
        Set<String> tmpFeatures = new LinkedHashSet<>();
        if (k == 0 || featureSet.featureNames().size() == 0) {
            throw new IllegalArgumentException("Tried to select zero features.");
        } else if (k != FeatureSelector.SELECT_ALL && !featureSet.isOrdered()) {
            throw new IllegalArgumentException("Tried to select the top " + k + " features from an unordered feature set.");
        } else if (k > featureSet.featureNames().size()) {
            throw new IllegalArgumentException("Tried to select more features than are available in feature set, requested " + k + ", found " + featureSet.featureNames().size());
        } else if (k > 0) {
            List<String> featureList = featureSet.featureNames();
            for (int i = 0; i < k; i++) {
                tmpFeatures.add(featureList.get(i));
            }
        } else if (k < -1) {
            throw new IllegalArgumentException("Supplied k " + k + " but only k == -1 or 1 < k < N} is allowed.");
        } else {
            tmpFeatures.addAll(featureSet.featureNames());
        }
        selectedFeatures = Collections.unmodifiableSet(tmpFeatures);

        // Check for feature set overlap with dataset
        FeatureMap wfm = dataset.getFeatureMap();
        Set<String> datasetFeatures = new HashSet<>(wfm.keySet());
        datasetFeatures.retainAll(selectedFeatures);
        if (datasetFeatures.size() == 0) {
            throw new IllegalArgumentException("The selected feature set had no overlap with the supplied dataset.");
        }

        int tmpNumExamplesRemoved = 0;
        // Generate feature subset examples, dropping any empty examples
        MutableFeatureMap featureMap = new MutableFeatureMap();
        MutableOutputInfo<T> outputInfo = dataset.getOutputFactory().generateInfo();
        List<Feature> features = new ArrayList<>();
        for (Example<T> ex : dataset) {
            features.clear();
            // Copy the example including the weight, output and metadata
            ArrayExample<T> copy = new ArrayExample<>(ex);
            for (Feature f : ex) {
                if (selectedFeatures.contains(f.getName())) {
                    // We'll definitely keep this example, so observe its feature values
                    featureMap.add(f.getName(), f.getValue());
                } else {
                    features.add(f);
                }
            }
            // Check if there are features to remove
            if (features.size() > 0) {
                copy.removeFeatures(features);
            }
            // If the example is still valid then add it, otherwise increment the removed counter
            if (copy.size() > 0) {
                data.add(copy);
                outputInfo.observe(ex.getOutput());
            } else {
                tmpNumExamplesRemoved++;
            }
        }
        numExamplesRemoved = tmpNumExamplesRemoved;

        // Rebuild feature and output maps
        this.featureIDMap = new ImmutableFeatureMap(featureMap);
        this.outputIDInfo = outputInfo.generateImmutableOutputInfo();

        if (numExamplesRemoved > 0) {
            logger.info(String.format("filtered out %d examples because they had zero features after the selected feature set was applied.", numExamplesRemoved));
        }
    }

    private SelectedFeatureDataset(DataProvenance provenance, OutputFactory<T> factory, String tribuoVersion,
                                   ImmutableFeatureMap fmap, ImmutableOutputInfo<T> outputInfo,
                                   List<Example<T>> examples, int k, SelectedFeatureSet featureSet,
                                   Set<String> selectedFeatures, int numExamplesRemoved) {
        super(provenance,factory,tribuoVersion,fmap,outputInfo,examples,false);
        this.k = k;
        this.selectedFeatures = Collections.unmodifiableSet(selectedFeatures);
        this.featureSet = featureSet;
        this.numExamplesRemoved = numExamplesRemoved;
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
    public static SelectedFeatureDataset<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        SelectedFeatureDatasetProto proto = message.unpack(SelectedFeatureDatasetProto.class);
        DatasetDataCarrier<?> carrier = DatasetDataCarrier.deserialize(proto.getMetadata());
        Class<?> outputClass = carrier.outputFactory().getUnknownOutput().getClass();
        FeatureMap fmap = carrier.featureDomain();
        List<Example<?>> examples = new ArrayList<>();
        int idx = 0;
        for (ExampleProto e : proto.getExamplesList()) {
            Example<?> example = Example.deserialize(e);
            if (example.getOutput().getClass().equals(outputClass)) {
                for (Feature f : example) {
                    if (fmap.get(f.getName()) == null) {
                        throw new IllegalStateException("Invalid protobuf, feature domain does not contain feature " + f.getName() + " present in example at idx " + idx);
                    }
                }
                examples.add(example);
            } else {
                throw new IllegalStateException("Invalid protobuf, expected all examples to have output class " + outputClass + ", but found " + example.getOutput().getClass() + " in example idx " + idx);
            }
            idx++;
        }
        if (!(fmap instanceof ImmutableFeatureMap)) {
            throw new IllegalStateException("Invalid protobuf, feature map was not immutable");
        }
        if (!(carrier.outputDomain() instanceof ImmutableOutputInfo)) {
            throw new IllegalStateException("Invalid protobuf, output info was not immutable");
        }
        int k = proto.getK();
        if ((k < 1) && (k != -1)) {
            throw new IllegalStateException("Invalid protobuf, k must be positive or -1, found " + k);
        }
        int numRemoved = proto.getNumExamplesRemoved();
        if (numRemoved < 0) {
            throw new IllegalStateException("Invalid protobuf, number of examples removed must be non-negative, found " + numRemoved);
        }
        SelectedFeatureSet featureSet = ProtoUtil.deserialize(proto.getFeatureSet());
        List<String> featureList = proto.getSelectedFeaturesList();
        Set<String> selectedFeatures = new LinkedHashSet<>(featureList);
        if (selectedFeatures.size() != featureList.size()) {
            throw new IllegalStateException("Invalid protobuf, selected features contained duplicates, features = " + featureList);
        }
        for (String s : selectedFeatures) {
            if (fmap.get(s) == null) {
                throw new IllegalStateException("Invalid protobuf, some selected features were not found in the feature domain.");
            }
        }
        return new SelectedFeatureDataset(carrier.provenance(), carrier.outputFactory(), carrier.tribuoVersion(),
                (ImmutableFeatureMap) fmap, (ImmutableOutputInfo) carrier.outputDomain(), examples,
                k, featureSet, selectedFeatures, numRemoved);
    }

    /**
     * The number of examples removed due to a lack of features.
     * @return The number of removed examples.
     */
    public int getNumExamplesRemoved() {
        return numExamplesRemoved;
    }

    /**
     * The number of features to use.
     * <p>
     * -1 signals that all features in the supplied feature set were used.
     * @return The number of features to use.
     */
    public int getK() {
        return k;
    }

    /**
     * The feature set.
     * @return The feature set.
     */
    public SelectedFeatureSet getFeatureSet() {
        return featureSet;
    }

    /**
     * The subset of the selected feature set.
     * @return The used subset of the selected feature set.
     */
    public Set<String> getSelectedFeatures() {
        return selectedFeatures;
    }

    @Override
    public DatasetProvenance getProvenance() {
        return new SelectedFeatureDatasetProvenance(this);
    }

    @Override
    public DatasetProto serialize() {
        SelectedFeatureDatasetProto.Builder datasetBuilder = SelectedFeatureDatasetProto.newBuilder();

        datasetBuilder.setMetadata(createDataCarrier(featureIDMap,outputIDInfo).serialize());
        for (Example<T> e : data) {
            datasetBuilder.addExamples(e.serialize());
        }
        datasetBuilder.setNumExamplesRemoved(numExamplesRemoved);
        datasetBuilder.setK(k);
        datasetBuilder.setFeatureSet(featureSet.serialize());
        datasetBuilder.addAllSelectedFeatures(selectedFeatures);

        DatasetProto.Builder builder = DatasetProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(SelectedFeatureDataset.class.getName());
        builder.setSerializedData(Any.pack(datasetBuilder.build()));

        return builder.build();
    }

    /**
     * Provenance for {@link SelectedFeatureDataset}.
     */
    public static final class SelectedFeatureDatasetProvenance extends DatasetProvenance {
        private static final long serialVersionUID = 1L;

        private static final String K = "k";
        private static final String FEATURE_SET_PROVENANCE = "feature-set-provenance";
        private static final String DATASET_PROVENANCE = "original-data-provenance";

        private final IntProvenance k;
        private final FeatureSetProvenance featureSetProvenance;
        private final DataProvenance datasetProvenance;

        <T extends Output<T>> SelectedFeatureDatasetProvenance(SelectedFeatureDataset<T> dataset) {
            super(dataset.sourceProvenance, new ListProvenance<>(), dataset);
            this.k = new IntProvenance(K,dataset.k);
            this.featureSetProvenance = dataset.featureSet.getProvenance();
            this.datasetProvenance = dataset.sourceProvenance;
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public SelectedFeatureDatasetProvenance(Map<String,Provenance> map) {
            super(map);
            this.k = ObjectProvenance.checkAndExtractProvenance(map,K,IntProvenance.class,SelectedFeatureDatasetProvenance.class.getSimpleName());
            this.featureSetProvenance = ObjectProvenance.checkAndExtractProvenance(map,FEATURE_SET_PROVENANCE, FeatureSetProvenance.class, SelectedFeatureDatasetProvenance.class.getSimpleName());
            this.datasetProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASET_PROVENANCE, DataProvenance.class, SelectedFeatureDatasetProvenance.class.getSimpleName());
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            if (!super.equals(o)) return false;
            SelectedFeatureDatasetProvenance pairs = (SelectedFeatureDatasetProvenance) o;
            return k.equals(pairs.k) && featureSetProvenance.equals(pairs.featureSetProvenance) && datasetProvenance.equals(pairs.datasetProvenance);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), k, featureSetProvenance, datasetProvenance);
        }

        @Override
        protected List<Pair<String, Provenance>> allProvenances() {
            List<Pair<String,Provenance>> provenances = super.allProvenances();
            provenances.add(new Pair<>(K,k));
            provenances.add(new Pair<>(FEATURE_SET_PROVENANCE,featureSetProvenance));
            provenances.add(new Pair<>(DATASET_PROVENANCE,datasetProvenance));
            return provenances;
        }
    }
}

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
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceException;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.DatasetProto;
import org.tribuo.protos.core.DatasetViewProto;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Predicate;

/**
 * DatasetView provides an immutable view on another {@link Dataset} that only exposes selected examples.
 * Does not copy the examples.
 *
 * @param <T> The output type of this dataset.
 */
public final class DatasetView<T extends Output<T>> extends ImmutableDataset<T> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final Dataset<T> innerDataset;

    private final int size;

    private final int[] exampleIndices;

    private final long seed;

    private final String tag;

    private final boolean sampled;

    private final boolean weighted;

    private boolean storeIndices = false;

    /**
     * Creates a DatasetView which includes the supplied indices from the dataset.
     * <p>
     * It uses the feature and output infos from the wrapped dataset.
     *
     * @param dataset The dataset to wrap.
     * @param exampleIndices The indices to present.
     * @param tag A tag for the view.
     */
    public DatasetView(Dataset<T> dataset, int[] exampleIndices, String tag) {
        this(dataset,exampleIndices,dataset.getFeatureIDMap(),dataset.getOutputIDInfo(), tag);
    }

    /**
     * Creates a DatasetView which includes the supplied indices from the dataset.
     * <p>
     * This takes the ImmutableFeatureMap and ImmutableOutputInfo parameters to save them being
     * regenerated (e.g., in BaggingTrainer).
     *
     * @param dataset The dataset to sample from.
     * @param exampleIndices The indices of this view in the wrapped dataset.
     * @param featureIDs The featureIDs to use for this dataset.
     * @param labelIDs The labelIDs to use for this dataset.
     * @param tag A tag for the view.
     */
    public DatasetView(Dataset<T> dataset, int[] exampleIndices, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> labelIDs, String tag) {
        super(dataset.getProvenance(),dataset.getOutputFactory(),featureIDs,labelIDs);
        if (!validateIndices(dataset.size(),exampleIndices)) {
            throw new IllegalArgumentException("Invalid indices supplied, dataset.size() = " + dataset.size() + ", but found a negative index or a value greater than or equal to size.");
        }
        this.innerDataset = dataset;
        this.size = exampleIndices.length;
        this.exampleIndices = exampleIndices;
        this.seed = -1;
        this.tag = tag;
        this.storeIndices = true;
        this.sampled = false;
        this.weighted = false;
    }

    /**
     * Constructor used by the sampling factory methods.
     * @param dataset The dataset to create the view over.
     * @param exampleIndices The indices to use.
     * @param seed The seed for the RNG.
     * @param featureIDs The feature IDs to use.
     * @param outputIDs The output IDs to use.
     * @param weighted Is it a weighted sample? (Weighted samples store the indices in the provenance by default).
     */
    private DatasetView(Dataset<T> dataset, int[] exampleIndices, long seed, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> outputIDs, boolean weighted) {
        super(dataset.getProvenance(),dataset.getOutputFactory(),featureIDs,outputIDs);
        this.innerDataset = dataset;
        this.size = exampleIndices.length;
        this.exampleIndices = exampleIndices;
        this.tag = "";
        this.seed = seed;
        this.sampled = true;
        this.weighted = weighted;
        this.storeIndices = weighted;
    }

    /**
     * Deserialization constructor.
     * @param dataset The dataset to create the view over.
     * @param exampleIndices The indices to use.
     * @param seed The seed for the RNG.
     * @param tag The dataset view tag.
     * @param featureIDs The feature IDs to use.
     * @param outputIDs The output IDs to use.
     * @param sampled Is it sampled?
     * @param weighted Is it a weighted sample?
     * @param storeIndices Should the indices be stored in the provenance?
     */
    private DatasetView(Dataset<T> dataset, int[] exampleIndices, long seed, String tag, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> outputIDs, boolean sampled, boolean weighted, boolean storeIndices) {
        super(dataset.getProvenance(),dataset.getOutputFactory(),featureIDs,outputIDs);
        this.innerDataset = dataset;
        this.size = exampleIndices.length;
        this.exampleIndices = exampleIndices;
        this.tag = tag;
        this.seed = seed;
        this.sampled = sampled;
        this.weighted = weighted;
        this.storeIndices = storeIndices;
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
    public static DatasetView<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        DatasetViewProto proto = message.unpack(DatasetViewProto.class);
        Dataset<?> inner = Dataset.deserialize(proto.getInnerDataset());
        Class<?> outputClass = inner.getOutputFactory().getUnknownOutput().getClass();
        OutputInfo<?> outputDomain = OutputInfo.deserialize(proto.getOutputDomain());
        Set<?> domain = outputDomain.getDomain();
        for (Object o : domain) {
            if (!o.getClass().equals(outputClass)) {
                throw new IllegalStateException("Invalid protobuf, output domains do not match, expected " + outputClass + " found " + o.getClass());
            }
        }
        FeatureMap featureDomain = FeatureMap.deserialize(proto.getFeatureDomain());
        int[] indices = Util.toPrimitiveInt(proto.getIndicesList());
        if (!validateIndices(inner.size(),indices)) {
            throw new IllegalStateException("Invalid protobuf, indices are not all inside the range of the inner dataset");
        }
        for (int i = 0; i < indices.length; i++) {
            Example<?> example = inner.getExample(indices[i]);
            for (Feature f : example) {
                if (featureDomain.get(f.getName()) == null) {
                    throw new IllegalStateException("Invalid protobuf, feature domain does not contain feature " + f.getName() + " present in an example");
                }
            }
        }
        if (!(featureDomain instanceof ImmutableFeatureMap)) {
            throw new IllegalStateException("Invalid protobuf, feature map was not immutable");
        }
        if (!(outputDomain instanceof ImmutableOutputInfo)) {
            throw new IllegalStateException("Invalid protobuf, output info was not immutable");
        }
        return new DatasetView(inner, indices, proto.getSeed(), proto.getTag(),
            (ImmutableFeatureMap) featureDomain, (ImmutableOutputInfo) outputDomain, proto.getSampled(),
            proto.getWeighted(), proto.getStoreIndices());
    }

    /**
     * Creates a view from the supplied dataset, using the specified predicate to
     * test if each example should be in this view.
     * @param dataset The dataset to create a view over.
     * @param predicate The predicate which determines if an example is in this view.
     * @param tag A tag denoting what the predicate does.
     * @param <T> The type of the Output in the dataset.
     * @return A dataset view containing each example where the predicate is true.
     */
    public static <T extends Output<T>> DatasetView<T> createView(Dataset<T> dataset, Predicate<Example<T>> predicate, String tag) {
        List<Integer> selectedIndices = new ArrayList<>();

        int i = 0;
        for (Example<T> e : dataset) {
            if (predicate.test(e)) {
                selectedIndices.add(i);
            }
            i++;
        }

        int[] exampleIndices = Util.toPrimitiveInt(selectedIndices);
        return new DatasetView<>(dataset,exampleIndices,tag);
    }

    /**
     * Generates a DatasetView bootstrapped from the supplied Dataset.
     *
     * @param dataset The dataset to sample from.
     * @param size The size of the sample.
     * @param seed A seed for the RNG.
     * @param <T> The type of the Output in the dataset.
     * @return A dataset view containing a bootstrap sample of the supplied dataset.
     */
    public static <T extends Output<T>> DatasetView<T> createBootstrapView(Dataset<T> dataset, int size, long seed) {
        return createBootstrapView(dataset,size,seed,dataset.getFeatureIDMap(),dataset.getOutputIDInfo());
    }

    /**
     * Generates a DatasetView bootstrapped from the supplied Dataset.
     * <p>
     * This takes the ImmutableFeatureMap and ImmutableOutputInfo parameters to save them being
     * regenerated.
     *
     * @param dataset The dataset to sample from.
     * @param size The size of the sample.
     * @param seed A seed for the RNG.
     * @param featureIDs The featureIDs to use for this dataset.
     * @param outputIDs The output info to use for this dataset.
     * @param <T> The type of the Output in the dataset.
     * @return A dataset view containing a bootstrap sample of the supplied dataset.
     */
    public static <T extends Output<T>> DatasetView<T> createBootstrapView(Dataset<T> dataset, int size, long seed, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> outputIDs) {
        int[] bootstrapIndices = Util.generateBootstrapIndices(size, dataset.size(), new SplittableRandom(seed));
        return new DatasetView<>(dataset, bootstrapIndices, seed, featureIDs, outputIDs, false);
    }

    /**
     * Generates a DatasetView bootstrapped from the supplied Dataset using the supplied
     * example weights.
     *
     * @param dataset The dataset to sample from.
     * @param size The size of the sample.
     * @param seed A seed for the RNG.
     * @param exampleWeights The sampling weights for each example, must be in the range 0,1.
     * @param <T> The type of the Output in the dataset.
     * @return A dataset view containing a weighted bootstrap sample of the supplied dataset.
     */
    public static <T extends Output<T>> DatasetView<T> createWeightedBootstrapView(Dataset<T> dataset, int size, long seed, float[] exampleWeights) {
        return createWeightedBootstrapView(dataset,size,seed,exampleWeights,dataset.getFeatureIDMap(),dataset.getOutputIDInfo());
    }

    /**
     * Generates a DatasetView bootstrapped from the supplied Dataset using the supplied
     * example weights.
     * <p>
     * This takes the ImmutableFeatureMap and ImmutableOutputInfo parameters to save them being
     * regenerated.
     *
     * @param dataset The dataset to sample from.
     * @param size The size of the sample.
     * @param seed A seed for the RNG.
     * @param exampleWeights The sampling weights for each example, must be in the range 0,1.
     * @param featureIDs The featureIDs to use for this dataset.
     * @param outputIDs The output info to use for this dataset.
     * @param <T> The type of the Output in the dataset.
     * @return A dataset view containing a weighted bootstrap sample of the supplied dataset.
     */
    public static <T extends Output<T>> DatasetView<T> createWeightedBootstrapView(Dataset<T> dataset, int size, long seed, float[] exampleWeights, ImmutableFeatureMap featureIDs, ImmutableOutputInfo<T> outputIDs) {
        if (dataset.size() != exampleWeights.length) {
            throw new IllegalArgumentException("There must be a weight for each example, dataset.size()="+dataset.size()+", exampleWeights.length="+exampleWeights.length);
        }
        int[] bootstrapIndices = Util.generateWeightedIndicesSample(size,exampleWeights,new SplittableRandom(seed));
        return new DatasetView<>(dataset, bootstrapIndices, seed, featureIDs, outputIDs,true);
    }

    /**
     * Are the indices stored in the provenance system.
     * @return True if the indices will be stored in the provenance of this view.
     */
    public boolean storeIndicesInProvenance() {
        return storeIndices;
    }

    /**
     * Set to true to store the indices in the provenance system.
     * @param storeIndices True if the indices should be stored in the provenance of this view.
     */
    public void setStoreIndices(boolean storeIndices) {
        this.storeIndices = storeIndices;
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append("DatasetView(innerDataset=");
        buffer.append(innerDataset.getSourceDescription());
        buffer.append(",size=");
        buffer.append(size);
        buffer.append(",seed=");
        buffer.append(seed);
        buffer.append(",tag=");
        buffer.append(tag);
        buffer.append(")");
        
        return buffer.toString();
    }

    /**
     * Gets the set of outputs that occur in the examples in this dataset.
     *
     * @return the set of outputs that occur in the examples in this dataset.
     */
    @Override
    public Set<T> getOutputs() {
        return innerDataset.getOutputs();
    }

    /**
     * Gets the size of the data set.
     *
     * @return the size of the data set.
     */
    @Override
    public int size() {
        return size;
    }

    @Override
    public ImmutableFeatureMap getFeatureMap() {
        return featureIDMap;
    }

    @Override
    public ImmutableOutputInfo<T> getOutputInfo() {
        return outputIDInfo;
    }

    @Override
    public Iterator<Example<T>> iterator() {
        return new ViewIterator<>(this);
    }

    @Override
    public List<Example<T>> getData() {
        ArrayList<Example<T>> data = new ArrayList<>();
        for (int index : exampleIndices) {
            data.add(innerDataset.getExample(index));
        }
        return Collections.unmodifiableList(data);
    }

    @Override
    public Example<T> getExample(int index) {
        if ((index < 0) || (index >= size())) {
            throw new IllegalArgumentException("Example index " + index + " is out of bounds.");  
        }
        return innerDataset.getExample(exampleIndices[index]);
    }

    @Override
    public DatasetViewProvenance getProvenance() {
        return new DatasetViewProvenance(this,storeIndices);
    }

    /**
     * The tag associated with this dataset, if it exists.
     * @return The dataset tag.
     */
    public String getTag() {
        return tag;
    }

    /**
     * Returns a copy of the indices used in this view.
     * @return The indices.
     */
    public int[] getExampleIndices() {
        return Arrays.copyOf(exampleIndices,exampleIndices.length);
    }

    @Override
    public DatasetProto serialize() {
        DatasetViewProto.Builder datasetBuilder = DatasetViewProto.newBuilder();

        datasetBuilder.setInnerDataset(innerDataset.serialize());
        datasetBuilder.setSize(size);
        for (int i = 0; i < exampleIndices.length; i++) {
            datasetBuilder.addIndices(exampleIndices[i]);
        }
        datasetBuilder.setSeed(seed);
        datasetBuilder.setTag(tag);
        datasetBuilder.setSampled(sampled);
        datasetBuilder.setWeighted(weighted);
        datasetBuilder.setStoreIndices(storeIndices);
        datasetBuilder.setFeatureDomain(featureIDMap.serialize());
        datasetBuilder.setOutputDomain(outputIDInfo.serialize());

        DatasetProto.Builder builder = DatasetProto.newBuilder();

        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(DatasetView.class.getName());
        builder.setSerializedData(Any.pack(datasetBuilder.build()));

        return builder.build();
    }

    /**
     * Checks that all the indices are non-negative and less than size.
     * @param size The maximum size.
     * @param indices The indices to check.
     * @return True if the indices are valid for the given size, false otherwise.
     */
    private static boolean validateIndices(int size, int[] indices) {
        boolean valid = true;

        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            valid &= idx < size && idx > -1;
        }

        return valid;
    }

    private static final class ViewIterator<T extends Output<T>> implements Iterator<Example<T>> {

        private int counter = 0;
        private final DatasetView<T> dataset;

        ViewIterator(DatasetView<T> dataset) {
            this.dataset = dataset;
        }

        @Override
        public boolean hasNext() {
            return counter < dataset.size();
        }

        @Override
        public Example<T> next() {
            Example<T> example = dataset.getExample(counter);
            counter++;
            return example;
        }

    }

    /**
     * Provenance for the {@link DatasetView}.
     */
    public static final class DatasetViewProvenance extends DatasetProvenance {
        private static final long serialVersionUID = 1L;

        private static final String SIZE = "size";
        private static final String SEED = "seed";
        private static final String TAG = "tag";
        private static final String SAMPLED = "sampled";
        private static final String WEIGHTED = "weighted";
        private static final String INDICES = "indices";

        private final IntProvenance size;
        private final LongProvenance seed;
        private final StringProvenance tag;
        private final BooleanProvenance weighted;
        private final BooleanProvenance sampled;
        private final int[] indices;

        <T extends Output<T>> DatasetViewProvenance(DatasetView<T> dataset, boolean storeIndices) {
            super(dataset.sourceProvenance, new ListProvenance<>(), dataset);
            this.size = new IntProvenance(SIZE,dataset.size);
            this.seed = new LongProvenance(SEED,dataset.seed);
            this.weighted = new BooleanProvenance(WEIGHTED,dataset.weighted);
            this.sampled = new BooleanProvenance(SAMPLED,dataset.sampled);
            this.tag = new StringProvenance(TAG,dataset.tag);
            this.indices = storeIndices ? dataset.indices : new int[0];
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public DatasetViewProvenance(Map<String,Provenance> map) {
            super(map);
            this.size = ObjectProvenance.checkAndExtractProvenance(map,SIZE,IntProvenance.class, DatasetViewProvenance.class.getSimpleName());
            this.seed = ObjectProvenance.checkAndExtractProvenance(map,SEED,LongProvenance.class, DatasetViewProvenance.class.getSimpleName());
            this.tag = ObjectProvenance.checkAndExtractProvenance(map,TAG,StringProvenance.class, DatasetViewProvenance.class.getSimpleName());
            this.weighted = ObjectProvenance.checkAndExtractProvenance(map,WEIGHTED,BooleanProvenance.class, DatasetViewProvenance.class.getSimpleName());
            this.sampled = ObjectProvenance.checkAndExtractProvenance(map,SAMPLED,BooleanProvenance.class, DatasetViewProvenance.class.getSimpleName());
            @SuppressWarnings("unchecked") // List provenance cast
            ListProvenance<IntProvenance> listIndices = ObjectProvenance.checkAndExtractProvenance(map,INDICES,ListProvenance.class, DatasetViewProvenance.class.getSimpleName());
            if (listIndices.getList().size() > 0) {
                try {
                    IntProvenance i = listIndices.getList().get(0);
                } catch (ClassCastException e) {
                    throw new ProvenanceException("Loaded another class when expecting an ListProvenance<IntProvenance>",e);
                }
            }
            this.indices = Util.toPrimitiveInt(ProvenanceUtil.unwrap(listIndices));
        }

        /**
         * Generates the indices from this DatasetViewProvenance
         * by rerunning the bootstrap sample.
         * <p>
         * Note these indices are invalid if the view is a weighted sample, or
         * not sampled.
         * @return The bootstrap indices.
         */
        public int[] generateBootstrap() {
            return Util.generateBootstrapIndices(size.getValue(), new SplittableRandom(seed.getValue()));
        }

        /**
         * Is this view from a bootstrap sample.
         * @return True if it's a bootstrap sample.
         */
        public boolean isSampled() {
            return sampled.getValue();
        }

        /**
         * Is this view a weighted bootstrap sample.
         * @return True if it's a weighted bootstrap sample.
         */
        public boolean isWeighted() {
            return weighted.getValue();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof DatasetView.DatasetViewProvenance)) return false;
            if (!super.equals(o)) return false;
            DatasetViewProvenance pairs = (DatasetViewProvenance) o;
            return size.equals(pairs.size) && seed.equals(pairs.seed) &&
                    tag.equals(pairs.tag);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), size, seed, tag);
        }

        @Override
        protected List<Pair<String, Provenance>> allProvenances() {
            List<Pair<String,Provenance>> provenances = super.allProvenances();
            provenances.add(new Pair<>(SIZE,size));
            provenances.add(new Pair<>(SEED,seed));
            provenances.add(new Pair<>(TAG,tag));
            provenances.add(new Pair<>(WEIGHTED,weighted));
            provenances.add(new Pair<>(SAMPLED,sampled));
            provenances.add(new Pair<>(INDICES,boxArray()));
            return provenances;
        }

        private ListProvenance<IntProvenance> boxArray() {
            List<IntProvenance> list = new ArrayList<>();

            for (int i = 0; i < indices.length; i++) {
                list.add(new IntProvenance("indices",indices[i]));
            }

            return new ListProvenance<>(list);
        }

        /**
         * This toString doesn't put the indices in the string, as it's likely
         * to be huge.
         * @return A string describing this provenance.
         */
        @Override
        public String toString() {
            List<Pair<String,Provenance>> provenances = super.allProvenances();
            provenances.add(new Pair<>(SIZE,size));
            provenances.add(new Pair<>(SEED,seed));
            provenances.add(new Pair<>(TAG,tag));
            provenances.add(new Pair<>(WEIGHTED,weighted));
            provenances.add(new Pair<>(SAMPLED,sampled));
            provenances.add(new Pair<>(INDICES,new ListProvenance<>()));

            StringBuilder sb = new StringBuilder();

            sb.append("DatasetView(");
            for (Pair<String,Provenance> p : provenances) {
                sb.append(p.getA());
                sb.append('=');
                sb.append(p.getB().toString());
                sb.append(',');
            }
            sb.replace(sb.length()-1,sb.length(),")");

            return sb.toString();
        }
    }
}

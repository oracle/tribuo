/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.transform;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.impl.ArrayExample;
import org.tribuo.protos.core.TransformerListProto;
import org.tribuo.protos.core.TransformerMapProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.transform.TransformerMap.TransformerMapProvenance;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A collection of {@link Transformer}s which can be applied to a {@link Dataset}
 * or {@link Example}. A TransformerMap is created by applying a {@link TransformationMap}
 * to a Dataset. It contains Transformers which are specific to the Dataset which created
 * it, for example the range of a feature used in binning is fixed to the value from
 * that Dataset.
 * <p>
 * Transformations only operate on observed values. To operate on implicit zeros then
 * first call {@link MutableDataset#densify} on the datasets.
 * See {@link org.tribuo.transform} for a more detailed discussion of densify.
 */
public final class TransformerMap implements ProtoSerializable<TransformerMapProto>,
        Provenancable<TransformerMapProvenance>, Serializable {
    
    private static final Logger logger = Logger.getLogger(TransformerMap.class.getName());
    
    private static final long serialVersionUID = 2L;

    private final Map<String, List<Transformer>> map;
    private final DatasetProvenance datasetProvenance;
    private final ConfiguredObjectProvenance transformationMapProvenance;

    /**
     * Constructs a transformer map which encapsulates a set of transformers that can be applied to features.
     * @param map The transformers, one per transformed feature.
     * @param datasetProvenance The provenance of the dataset the transformers were fit against.
     * @param transformationMapProvenance The provenance of the transformation map that was fit.
     */
    public TransformerMap(Map<String,List<Transformer>> map, DatasetProvenance datasetProvenance, ConfiguredObjectProvenance transformationMapProvenance) {
        this.map = Collections.unmodifiableMap(map);
        this.datasetProvenance = datasetProvenance;
        this.transformationMapProvenance = transformationMapProvenance;
    }

    /**
     * Deserializes a {@link TransformerMapProto} into a {@link TransformerMap}.
     * @param proto The proto to deserialize.
     * @return The deserialized TransformerMap.
     */
    public static TransformerMap deserialize(TransformerMapProto proto) {
        if (proto.getVersion() == 0) {
            Map<String,List<Transformer>> map = new LinkedHashMap<>();
            for (Map.Entry<String,TransformerListProto> e : proto.getTransformersMap().entrySet()) {
                List<Transformer> list = new ArrayList<>();
                for (TransformerProto p : e.getValue().getTransformerList()) {
                    list.add(ProtoUtil.deserialize(p));
                }
                map.put(e.getKey(),list);
            }
            DatasetProvenance datasetProvenance = (DatasetProvenance) ProvenanceUtil.unmarshalProvenance(
                    PROVENANCE_SERIALIZER.deserializeFromProto(proto.getDatasetProvenance()));
            ConfiguredObjectProvenance transformationMapProvenance = (ConfiguredObjectProvenance)
                    ProvenanceUtil.unmarshalProvenance(PROVENANCE_SERIALIZER.deserializeFromProto(
                            proto.getTransformationMapProvenance()));
            return new TransformerMap(map,datasetProvenance,transformationMapProvenance);
        } else {
            throw new IllegalArgumentException("Unknown version " + proto.getVersion() + " expected {0}");
        }
    }

    /**
     * Applies a {@link List} of {@link Transformer}s to the supplied double value,
     * returning the transformed value.
     * @param value The value to transform.
     * @param transformerList The transformers to apply.
     * @return The transformed value.
     */
    public static double applyTransformerList(double value, List<Transformer> transformerList) {
        if (transformerList != null) {
            for (Transformer t : transformerList) {
                value = t.transform(value);
            }
        }
        return value;
    }

    /**
     * Copies the supplied example and applies the transformers to it.
     * @param example The example to transform.
     * @param <T> The type of Output.
     * @return A copy of the example with the transformers applied to it's features.
     */
    public <T extends Output<T>> Example<T> transformExample(Example<T> example) {
        ArrayExample<T> newExample = new ArrayExample<>(example);
        newExample.transform(this);
        return newExample;
    }

    /**
     * Copies the supplied example and applies the transformers to it.
     * @param example The example to transform.
     * @param featureNames The feature names to densify.
     * @param <T> The type of Output.
     * @return A copy of the example with the transformers applied to it's features.
     */
    public <T extends Output<T>> Example<T> transformExample(Example<T> example, List<String> featureNames) {
        ArrayExample<T> newExample = new ArrayExample<>(example);
        newExample.densify(featureNames);
        newExample.transform(this);
        return newExample;
    }

    /**
     * Copies the supplied dataset and applies the transformers to each example in it.
     * <p>
     * Does not densify the dataset first.
     * @param dataset The dataset to transform.
     * @param <T> The type of Output.
     * @return A deep copy of the dataset (and it's examples) with the transformers applied to it's features.
     */
    public <T extends Output<T>> MutableDataset<T> transformDataset(Dataset<T> dataset) {
        return transformDataset(dataset,false);
    }

    /**
     * Copies the supplied dataset and applies the transformers to each example in it.
     * @param dataset The dataset to transform.
     * @param densify Densify the dataset before transforming it.
     * @param <T> The type of Output.
     * @return A deep copy of the dataset (and it's examples) with the transformers applied to it's features.
     */
    public <T extends Output<T>> MutableDataset<T> transformDataset(Dataset<T> dataset, boolean densify) {
        
        logger.fine("Creating deep copy of data set");
        
        MutableDataset<T> newDataset = MutableDataset.createDeepCopy(dataset);

        if (densify) {
            newDataset.densify();
        }

        logger.fine(String.format("Transforming data set copy"));
        
        newDataset.transform(this);

        return newDataset;
    }
    
    /**
     * Gets the size of the map.
     * 
     * @return the size of the map of feature names to transformers.
     */
    public int size() {
        return map.size();
    }
    
    /**
     * Gets the transformers associated with a given feature name.
     * @param featureName the name of the feature for which we want the transformer
     * @return the transformer list associated with the feature name, which may be <code>null</code>
     * if there is no feature with that name.
     */
    public List<Transformer> get(String featureName) {
        return map.get(featureName);
    }

    @Override
    public String toString() {
        return "TransformerMap(map="+map.toString()+")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TransformerMap that = (TransformerMap) o;
        return map.equals(that.map) && datasetProvenance.equals(that.datasetProvenance) && transformationMapProvenance.equals(that.transformationMapProvenance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(map, datasetProvenance, transformationMapProvenance);
    }

    /**
     * Get the feature names and associated list of transformers.
     * @return The entry set of the transformer map.
     */
    public Set<Map.Entry<String,List<Transformer>>> entrySet() {
        return map.entrySet();
    }

    @Override
    public TransformerMapProvenance getProvenance() {
        return new TransformerMapProvenance(this);
    }

    @Override
    public TransformerMapProto serialize() {
        TransformerMapProto.Builder builder = TransformerMapProto.newBuilder();

        builder.setVersion(0);
        builder.setDatasetProvenance(
                PROVENANCE_SERIALIZER.serializeToProto(
                        ProvenanceUtil.marshalProvenance(datasetProvenance)));
        builder.setTransformationMapProvenance(
                PROVENANCE_SERIALIZER.serializeToProto(
                        ProvenanceUtil.marshalProvenance(transformationMapProvenance)));
        for (Map.Entry<String, List<Transformer>> e : map.entrySet()) {
            TransformerListProto.Builder listBuilder = TransformerListProto.newBuilder();
            for (Transformer t : e.getValue()) {
                listBuilder.addTransformer(t.serialize());
            }
            builder.putTransformers(e.getKey(),listBuilder.build());
        }

        return builder.build();
    }

    /**
     * Provenance for {@link TransformerMap}.
     */
    public final static class TransformerMapProvenance implements ObjectProvenance {
        private static final long serialVersionUID = 1L;

        private static final String TRANSFORMATION_MAP = "transformation-map";
        private static final String DATASET = "dataset";

        private final String className;
        private final ConfiguredObjectProvenance transformationMapProvenance;
        private final DatasetProvenance datasetProvenance;

        TransformerMapProvenance(TransformerMap host) {
            this.className = host.getClass().getName();
            this.transformationMapProvenance = host.transformationMapProvenance;
            this.datasetProvenance = host.datasetProvenance;
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public TransformerMapProvenance(Map<String,Provenance> map) {
            this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class,TransformerMapProvenance.class.getSimpleName()).getValue();
            this.transformationMapProvenance = ObjectProvenance.checkAndExtractProvenance(map,TRANSFORMATION_MAP,ConfiguredObjectProvenance.class,TransformerMapProvenance.class.getSimpleName());
            this.datasetProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASET,DatasetProvenance.class,TransformerMapProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return className;
        }

        @Override
        public Iterator<Pair<String, Provenance>> iterator() {
            ArrayList<Pair<String,Provenance>> list = new ArrayList<>();

            list.add(new Pair<>(CLASS_NAME,new StringProvenance(CLASS_NAME,className)));
            list.add(new Pair<>(TRANSFORMATION_MAP,transformationMapProvenance));
            list.add(new Pair<>(DATASET,datasetProvenance));

            return list.iterator();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof TransformerMapProvenance)) return false;
            TransformerMapProvenance pairs = (TransformerMapProvenance) o;
            return className.equals(pairs.className) &&
                    transformationMapProvenance.equals(pairs.transformationMapProvenance) &&
                    datasetProvenance.equals(pairs.datasetProvenance);
        }

        @Override
        public int hashCode() {
            return Objects.hash(className, transformationMapProvenance, datasetProvenance);
        }

        @Override
        public String toString() {
            return generateString("TransformerMap");
        }
    }
}

/*
 * Copyright (c) 2015, 2023, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.protos.core.TransformedModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Wraps a {@link Model} with it's {@link TransformerMap} so all {@link Example}s are transformed
 * appropriately before the model makes predictions.
 * <p>
 * If the densify flag is set, densifies all incoming data before transforming it.
 * <p>
 * Transformations only operate on observed values. To operate on implicit zeros then
 * first call {@link MutableDataset#densify} on the datasets.
 */
public class TransformedModel<T extends Output<T>> extends Model<T> {

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final Model<T> innerModel;

    private final TransformerMap transformerMap;

    private final boolean densify;

    private final ArrayList<String> featureNames;

    TransformedModel(ModelProvenance modelProvenance, Model<T> innerModel, TransformerMap transformerMap, boolean densify) {
        this(innerModel.getName(), modelProvenance, innerModel, transformerMap, densify);
    }

    private TransformedModel(String name, ModelProvenance modelProvenance, Model<T> innerModel, TransformerMap transformerMap, boolean densify) {
        super(name,
              modelProvenance,
              innerModel.getFeatureIDMap(),
              innerModel.getOutputIDInfo(),
              innerModel.generatesProbabilities());
        this.innerModel = innerModel;
        this.transformerMap = transformerMap;
        this.densify = densify;
        this.featureNames = new ArrayList<>(featureIDMap.keySet());
        Collections.sort(featureNames);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"unchecked","rawtypes"})
    public static TransformedModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        TransformedModelProto proto = message.unpack(TransformedModelProto.class);

        // We discard the output domain and feature domain from the carrier and use the ones in the inner model.
        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        Model<?> model = Model.deserialize(proto.getModel());
        TransformerMap transformerMap = TransformerMap.deserialize(proto.getTransformerMap());

        return new TransformedModel(carrier.name(), carrier.provenance(), model, transformerMap, proto.getDensify());
    }

    /**
     * Gets the transformers that this model applies to each example.
     * <p>
     * Note if you use these transformers to modify some data, and then
     * feed that data to this model, the data will be transformed twice
     * and this is not what you want.
     * @return The transformers.
     */
    public TransformerMap getTransformerMap() {
        return transformerMap;
    }

    /**
     * Gets the inner model to allow access to any class specific methods
     * that model contains (e.g., to examine cluster centroids).
     * <p>
     * Note that this model expects all examples to have been transformed using
     * the transformer map, which can be extracted with {@link #getTransformerMap}.
     * @return The inner model.
     */
    public Model<T> getInnerModel() {
        return innerModel;
    }

    /**
     * Returns true if the model densifies the feature space before applying the transformations.
     * @return True if the transforms operate on the dense feature space.
     */
    public boolean getDensify() {
        return densify;
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        Example<T> transformedExample;
        if (densify) {
            transformedExample = transformerMap.transformExample(example,featureNames);
        } else {
            transformedExample = transformerMap.transformExample(example);
        }
        return innerModel.predict(transformedExample);
    }

    @Override
    public List<Prediction<T>> predict(Dataset<T> examples) {
        Dataset<T> transformedDataset = transformerMap.transformDataset(examples,densify);

        List<Prediction<T>> predictions = new ArrayList<>();
        for (Example<T> example : transformedDataset) {
            predictions.add(innerModel.predict(example));
        }

        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return innerModel.getTopFeatures(n);
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        Example<T> transformedExample = transformerMap.transformExample(example);
        return innerModel.getExcuse(transformedExample);
    }

    @Override
    protected TransformedModel<T> copy(String name, ModelProvenance newProvenance) {
        return new TransformedModel<>(newProvenance,innerModel,transformerMap,densify);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        TransformedModelProto.Builder modelBuilder = TransformedModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setModel(innerModel.serialize());
        modelBuilder.setTransformerMap(transformerMap.serialize());
        modelBuilder.setDensify(densify);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(TransformedModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }
}

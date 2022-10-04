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

package org.tribuo.interop.tensorflow;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.interop.tensorflow.protos.TensorFlowNativeModelProto;
import org.tribuo.interop.tensorflow.protos.TensorTupleProto;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * This model encapsulates a TensorFlow model running in graph mode with a single tensor output.
 * <p>
 * It accepts an {@link FeatureConverter} that converts an example's features into a set of {@link Tensor}s, and an
 * {@link OutputConverter} that converts a {@link Tensor} into a {@link Prediction}.
 * <p>
 * This model's serialized form stores the weights and is entirely self contained.
 * If you wish to convert it into a model which uses checkpoints then call {@link #convertToCheckpointModel}.
 * <p>
 * The model's serialVersionUID is set to the major TensorFlow version number times 100.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public final class TensorFlowNativeModel<T extends Output<T>> extends TensorFlowModel<T> {
    private static final long serialVersionUID = 200L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    TensorFlowNativeModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap,
                          ImmutableOutputInfo<T> outputIDMap, GraphDef trainedGraphDef,
                          Map<String, TensorFlowUtil.TensorTuple> tensorMap, int batchSize, String outputName,
                          FeatureConverter featureConverter, OutputConverter<T> outputConverter) {
        super(name, description, featureIDMap, outputIDMap, trainedGraphDef, batchSize, outputName, featureConverter, outputConverter);
        TensorFlowUtil.restoreMarshalledVariables(session,tensorMap);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"rawtypes","unchecked"}) // guarded by a getClass check that the output domain and converter are compatible
    public static TensorFlowNativeModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        TensorFlowNativeModelProto proto = message.unpack(TensorFlowNativeModelProto.class);

        OutputConverter<?> outputConverter = ProtoUtil.deserialize(proto.getOutputConverter());
        FeatureConverter featureConverter = ProtoUtil.deserialize(proto.getFeatureConverter());

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(outputConverter.getTypeWitness())) {
            throw new IllegalStateException("Invalid protobuf, output domain does not match converter, found " + carrier.outputDomain().getClass() + " and " + outputConverter.getTypeWitness());
        }

        GraphDef graphDef = GraphDef.parseFrom(proto.getModelDef());

        Map<String, TensorFlowUtil.TensorTuple> tensorMap = new HashMap<>();
        for (Map.Entry<String, TensorTupleProto> e : proto.getTensorsMap().entrySet()) {
            tensorMap.put(e.getKey(), new TensorFlowUtil.TensorTuple(e.getValue()));
        }

        return new TensorFlowNativeModel(carrier.name(), carrier.provenance(), carrier.featureDomain(),
                carrier.outputDomain(), graphDef, tensorMap, proto.getBatchSize(), proto.getOutputName(),
                featureConverter, outputConverter);
    }

    @Override
    protected TensorFlowNativeModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new TensorFlowNativeModel<>(newName,newProvenance,featureIDMap,outputIDInfo,modelGraph.toGraphDef(), TensorFlowUtil.extractMarshalledVariables(modelGraph,session),batchSize,outputName, featureConverter, outputConverter);
    }

    /**
     * Creates a {@link TensorFlowCheckpointModel} version of this model.
     * @param checkpointDirectory The directory to write the checkpoint to.
     * @param checkpointName The name of the checkpoint files.
     * @return A version of this model using a TensorFlow checkpoint to store the parameters.
     */
    public TensorFlowCheckpointModel<T> convertToCheckpointModel(String checkpointDirectory, String checkpointName) {
        session.save(Paths.get(checkpointDirectory,checkpointName).toString());
        return new TensorFlowCheckpointModel<>(name, provenance, featureIDMap,
                outputIDInfo, modelGraph.toGraphDef(), checkpointDirectory, checkpointName, batchSize, outputName, featureConverter, outputConverter);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();
        Map<String, TensorTupleProto> tensors = new HashMap<>();
        for (Map.Entry<String, TensorFlowUtil.TensorTuple> e : TensorFlowUtil.extractMarshalledVariables(modelGraph, session).entrySet()) {
            tensors.put(e.getKey(), e.getValue().serialize());
        }

        TensorFlowNativeModelProto.Builder modelBuilder = TensorFlowNativeModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setModelDef(ByteString.copyFrom(modelGraph.toGraphDef().toByteArray()));
        modelBuilder.putAllTensors(tensors);
        modelBuilder.setOutputName(outputName);
        modelBuilder.setBatchSize(batchSize);
        modelBuilder.setOutputConverter(outputConverter.serialize());
        modelBuilder.setFeatureConverter(featureConverter.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(TensorFlowNativeModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        if (closed) {
            throw new IllegalStateException("Can't serialize a closed model, the state has gone.");
        }
        out.defaultWriteObject();
        byte[] modelBytes = modelGraph.toGraphDef().toByteArray();
        out.writeObject(modelBytes);
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = TensorFlowUtil.extractMarshalledVariables(modelGraph, session);
        out.writeObject(tensorMap);
    }

    @SuppressWarnings("unchecked") //deserialising a typed map
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = (Map<String, TensorFlowUtil.TensorTuple>) in.readObject();
        modelGraph = new Graph();
        modelGraph.importGraphDef(GraphDef.parseFrom(modelBytes));
        session = new Session(modelGraph);
        TensorFlowUtil.restoreMarshalledVariables(session,tensorMap);
    }
}

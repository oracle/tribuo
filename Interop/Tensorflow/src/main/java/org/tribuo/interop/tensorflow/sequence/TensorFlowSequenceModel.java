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

package org.tribuo.interop.tensorflow.sequence;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.interop.tensorflow.TensorMap;
import org.tribuo.interop.tensorflow.TensorFlowUtil;
import org.tribuo.interop.tensorflow.protos.TensorFlowSequenceModelProto;
import org.tribuo.interop.tensorflow.protos.TensorTupleProto;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.sequence.SequenceExample;
import org.tribuo.sequence.SequenceModel;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A TensorFlow model which implements SequenceModel, suitable for use in sequential prediction tasks.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public class TensorFlowSequenceModel<T extends Output<T>> extends SequenceModel<T> implements AutoCloseable {

    private static final long serialVersionUID = 200L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private transient Graph modelGraph = null;
    private transient Session session = null;

    protected final SequenceFeatureConverter featureConverter;
    protected final SequenceOutputConverter<T> outputConverter;

    protected final String predictOp;

    TensorFlowSequenceModel(String name,
                            ModelProvenance description,
                            ImmutableFeatureMap featureIDMap,
                            ImmutableOutputInfo<T> outputIDMap,
                            GraphDef graphDef,
                            SequenceFeatureConverter featureConverter,
                            SequenceOutputConverter<T> outputConverter,
                            String predictOp,
                            Map<String, TensorFlowUtil.TensorTuple> tensorMap
    ) {
        super(name, description, featureIDMap, outputIDMap);
        this.featureConverter = featureConverter;
        this.outputConverter = outputConverter;
        this.predictOp = predictOp;
        this.modelGraph = new Graph();
        this.modelGraph.importGraphDef(graphDef);
        this.session = new Session(modelGraph);

        TensorFlowUtil.restoreMarshalledVariables(session, tensorMap);
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
    public static TensorFlowSequenceModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        TensorFlowSequenceModelProto proto = message.unpack(TensorFlowSequenceModelProto.class);

        SequenceOutputConverter<?> outputConverter = ProtoUtil.deserialize(proto.getOutputConverter());
        SequenceFeatureConverter featureConverter = ProtoUtil.deserialize(proto.getFeatureConverter());

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(outputConverter.getTypeWitness())) {
            throw new IllegalStateException("Invalid protobuf, output domain does not match converter, found " + carrier.outputDomain().getClass() + " and " + outputConverter.getTypeWitness());
        }

        GraphDef graphDef = GraphDef.parseFrom(proto.getModelDef());

        Map<String, TensorFlowUtil.TensorTuple> tensorMap = new HashMap<>();
        for (Map.Entry<String, TensorTupleProto> e : proto.getTensorsMap().entrySet()) {
            tensorMap.put(e.getKey(), new TensorFlowUtil.TensorTuple(e.getValue()));
        }

        return new TensorFlowSequenceModel(carrier.name(), carrier.provenance(), carrier.featureDomain(),
                carrier.outputDomain(), graphDef, featureConverter, outputConverter, proto.getPredictOp(),
                tensorMap);
    }

    @Override
    public List<Prediction<T>> predict(SequenceExample<T> example) {
        try (TensorMap feed = featureConverter.encode(example, featureIDMap)) {
            Session.Runner runner = session.runner();
            runner = feed.feedInto(runner);
            try (Tensor outputTensor = runner
                    .fetch(predictOp)
                    .run()
                    .get(0)) {
                return outputConverter.decode(outputTensor, example, outputIDMap);
            }
        }
    }

    /**
     * Returns an empty map, as the top features are not well defined for most TensorFlow models.
     */
    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int i) {
        return Collections.emptyMap();
    }

    /**
     * Close the session and graph if they exist.
     */
    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
        if (modelGraph != null) {
            modelGraph.close();
        }
    }

    @Override
    public SequenceModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();
        Map<String, TensorTupleProto> tensors = new HashMap<>();
        for (Map.Entry<String, TensorFlowUtil.TensorTuple> e : TensorFlowUtil.extractMarshalledVariables(modelGraph, session).entrySet()) {
            tensors.put(e.getKey(), e.getValue().serialize());
        }

        TensorFlowSequenceModelProto.Builder modelBuilder = TensorFlowSequenceModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setModelDef(ByteString.copyFrom(modelGraph.toGraphDef().toByteArray()));
        modelBuilder.putAllTensors(tensors);
        modelBuilder.setPredictOp(predictOp);
        modelBuilder.setOutputConverter(outputConverter.serialize());
        modelBuilder.setFeatureConverter(featureConverter.serialize());

        SequenceModelProto.Builder builder = SequenceModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(TensorFlowSequenceModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
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
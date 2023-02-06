/*
 * Copyright (c) 2021, 2023, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.IndependentSequenceModelProto;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A SequenceModel which independently predicts each element of the sequence.
 * @param <T> The output type.
 */
public class IndependentSequenceModel<T extends Output<T>> extends SequenceModel<T> {
    private static final Logger logger = Logger.getLogger(IndependentSequenceModel.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final Model<T> model;

    IndependentSequenceModel(String name, ModelProvenance description, Model<T> model) {
        super(name, description, model.getFeatureIDMap(), model.getOutputIDInfo());
        this.model = model;
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
    public static IndependentSequenceModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        IndependentSequenceModelProto proto = message.unpack(IndependentSequenceModelProto.class);

        // We discard the output domain and feature domain from the carrier and use the ones in the inner model.
        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        Model<?> model = Model.deserialize(proto.getModel());

        return new IndependentSequenceModel(carrier.name(), carrier.provenance(), model);
    }

    @Override
    public List<Prediction<T>> predict(SequenceExample<T> example) {
        List<Prediction<T>> output = new ArrayList<>();
        for (Example<T> e : example) {
            output.add(model.predict(e));
        }
        return output;
    }

    @Override
    public SequenceModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        IndependentSequenceModelProto.Builder modelBuilder = IndependentSequenceModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setModel(model.serialize());

        SequenceModelProto.Builder builder = SequenceModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(IndependentSequenceModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return model.getTopFeatures(n);
    }
}

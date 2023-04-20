/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.test;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.test.protos.MockModelProto;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A model which performs dummy classifications (e.g., constant output, uniform sampled labels, stratified sampled labels).
 */
public final class MockModel extends Model<MockOutput> {

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final MockOutput constantMockOutput;

    MockModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<MockOutput> outputIDInfo, MockOutput constantMockOutput) {
        super(name, description, featureIDMap, outputIDInfo, false);
        this.constantMockOutput = constantMockOutput;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MockModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        MockModelProto proto = message.unpack(MockModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(MockOutput.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<MockOutput> outputDomain = (ImmutableOutputInfo<MockOutput>) carrier.outputDomain();

        MockOutput constantMockOutput = new MockOutput(proto.getConstantOutput());

        return new MockModel(carrier.name(), carrier.provenance(), carrier.featureDomain(), outputDomain,
            constantMockOutput);
    }

    @Override
    public Prediction<MockOutput> predict(Example<MockOutput> example) {
        return new Prediction<>(constantMockOutput,0,example);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        Map<String,List<Pair<String,Double>>> map = new HashMap<>();
        if (n != 0) {
            map.put(Model.ALL_OUTPUTS, Collections.singletonList(new Pair<>(BIAS_FEATURE, 1.0)));
        }
        return map;
    }

    @Override
    public Optional<Excuse<MockOutput>> getExcuse(Example<MockOutput> example) {
        return Optional.of(new Excuse<>(example,predict(example),getTopFeatures(1)));
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<MockOutput> carrier = createDataCarrier();

        MockModelProto.Builder modelBuilder = MockModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setConstantOutput(constantMockOutput.label);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(MockModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    protected MockModel copy(String newName, ModelProvenance newProvenance) {
        return new MockModel(newName,newProvenance,featureIDMap,outputIDInfo,constantMockOutput.copy());
    }
}

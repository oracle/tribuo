/*
 * Copyright (c) 2022, 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.impl;

import com.oracle.labs.mlrg.olcut.config.protobuf.ProtoProvenanceSerialization;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputInfo;
import org.tribuo.protos.ProtoDeserializationCache;
import org.tribuo.protos.core.ModelDataProto;
import org.tribuo.provenance.ModelProvenance;

/**
 * Serialization carrier for common fields in Model and SequenceModel.
 *
 * @param name                   The model's name.
 * @param provenance             The model provenance.
 * @param featureDomain          The features this model knows about.
 * @param outputDomain           The outputs this model predicts.
 * @param generatesProbabilities Does this model generate probability distributions in the output.
 * @param tribuoVersion          The Tribuo version string.
 */
public record ModelDataCarrier<T extends Output<T>>(String name, ModelProvenance provenance,
                                                    ImmutableFeatureMap featureDomain,
                                                    ImmutableOutputInfo<T> outputDomain, boolean generatesProbabilities,
                                                    String tribuoVersion) {
    private static final ProtoProvenanceSerialization PROVENANCE_SERIALIZER = new ProtoProvenanceSerialization(false);

    /**
     * Deserializes a {@link ModelDataProto} into a {@link ModelDataCarrier}.
     * <p>
     * Uses an empty deserialization cache.
     *
     * @param proto The proto to deserialize.
     * @return The model data.
     * @deprecated The serialization cache version should be preferred.
     */
    @Deprecated
    public static ModelDataCarrier<?> deserialize(ModelDataProto proto) {
        return deserialize(proto, new ProtoDeserializationCache());
    }

    /**
     * Deserializes a {@link ModelDataProto} into a {@link ModelDataCarrier}.
     *
     * @param proto The proto to deserialize.
     * @param deserCache The current deserialization cache.
     * @return The model data.
     */
    public static ModelDataCarrier<?> deserialize(ModelDataProto proto, ProtoDeserializationCache deserCache) {
        ModelProvenance provenance = (ModelProvenance) ProvenanceUtil.unmarshalProvenance(PROVENANCE_SERIALIZER.deserializeFromProto(proto.getProvenance()));
        ImmutableFeatureMap featureDomain = (ImmutableFeatureMap) FeatureMap.deserialize(proto.getFeatureDomain(), deserCache);
        ImmutableOutputInfo<?> outputDomain = (ImmutableOutputInfo<?>) OutputInfo.deserialize(proto.getOutputDomain(), deserCache);
        return new ModelDataCarrier<>(proto.getName(), provenance, featureDomain, outputDomain, proto.getGenerateProbabilities(), proto.getTribuoVersion());
    }

    /**
     * Serializes this object to a protobuf.
     *
     * @return The protobuf representation.
     */
    public ModelDataProto serialize() {
        ModelDataProto.Builder builder = ModelDataProto.newBuilder();

        builder.setName(name);
        builder.setGenerateProbabilities(generatesProbabilities);
        builder.setFeatureDomain(featureDomain.serialize());
        builder.setOutputDomain(outputDomain.serialize());
        builder.setProvenance(PROVENANCE_SERIALIZER.serializeToProto(ProvenanceUtil.marshalProvenance(provenance)));
        builder.setTribuoVersion(tribuoVersion);

        return builder.build();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ModelDataCarrier<?> that)) {
            return false;
        }
        return generatesProbabilities == that.generatesProbabilities && name.equals(that.name)
                && provenance.equals(that.provenance) && featureDomain.equals(that.featureDomain)
                && outputDomain.equals(that.outputDomain) && tribuoVersion.equals(that.tribuoVersion);
    }

    @Override
    public String toString() {
        return "ModelDataCarrier{" +
                "name='" + name + '\'' +
                ", provenance=" + provenance +
                ", featureDomain=" + featureDomain +
                ", outputDomain=" + outputDomain +
                ", generatesProbabilities=" + generatesProbabilities +
                ", tribuoVersion=" + tribuoVersion +
                '}';
    }
}

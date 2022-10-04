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

package org.tribuo.impl;

import com.oracle.labs.mlrg.olcut.config.protobuf.ProtoProvenanceSerialization;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.ModelDataProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.Objects;

/**
 * Serialization carrier for common fields in Model and SequenceModel.
 * <p>
 * Likely to be a record one day.
 */
public final class ModelDataCarrier<T extends Output<T>> {
    private static final ProtoProvenanceSerialization PROVENANCE_SERIALIZER = new ProtoProvenanceSerialization(false);

    /**
     * The model's name.
     */
    private final String name;

    /**
     * The model provenance.
     */
    private final ModelProvenance provenance;

    /**
     * The features this model knows about.
     */
    private final ImmutableFeatureMap featureDomain;

    /**
     * The outputs this model predicts.
     */
    private final ImmutableOutputInfo<T> outputDomain;

    /**
     * Does this model generate probability distributions in the output.
     */
    private final boolean generatesProbabilities;

    /**
     * The Tribuo version string.
     */
    private final String tribuoVersion;

    /**
     * Constructs a new ModelDataCarrier.
     * <p>
     * Will be the canonical constructor for the record form.
     *
     * @param name                   The model name.
     * @param provenance             The model provenance.
     * @param featureDomain          The feature domain.
     * @param outputDomain           The output domain.
     * @param generatesProbabilities Does this model generate probabilities?
     * @param tribuoVersion          The Tribuo version string.
     */
    public ModelDataCarrier(String name, ModelProvenance provenance, ImmutableFeatureMap featureDomain, ImmutableOutputInfo<T> outputDomain, boolean generatesProbabilities, String tribuoVersion) {
        this.name = name;
        this.provenance = provenance;
        this.featureDomain = featureDomain;
        this.outputDomain = outputDomain;
        this.generatesProbabilities = generatesProbabilities;
        this.tribuoVersion = tribuoVersion;
    }

    /**
     * The model name.
     *
     * @return The model name.
     */
    public String name() {
        return name;
    }

    /**
     * The model provenance.
     *
     * @return The model provenance.
     */
    public ModelProvenance provenance() {
        return provenance;
    }

    /**
     * The feature domain.
     *
     * @return The feature domain.
     */
    public ImmutableFeatureMap featureDomain() {
        return featureDomain;
    }

    /**
     * The output domain.
     *
     * @return The output domain.
     */
    public ImmutableOutputInfo<T> outputDomain() {
        return outputDomain;
    }

    /**
     * Does this model generate probabilities?
     *
     * @return Does the model generate probabilities?
     */
    public boolean generatesProbabilities() {
        return generatesProbabilities;
    }

    /**
     * Gets the Tribuo version string.
     *
     * @return The Tribuo version string.
     */
    public String tribuoVersion() {
        return tribuoVersion;
    }

    /**
     * Deserializes a {@link ModelDataProto} into a {@link ModelDataCarrier}.
     *
     * @param proto The proto to deserialize.
     * @return The model data.
     */
    public static ModelDataCarrier<?> deserialize(ModelDataProto proto) {
        ModelProvenance provenance = (ModelProvenance) ProvenanceUtil.unmarshalProvenance(PROVENANCE_SERIALIZER.deserializeFromProto(proto.getProvenance()));
        ImmutableFeatureMap featureDomain = (ImmutableFeatureMap) FeatureMap.deserialize(proto.getFeatureDomain());
        ImmutableOutputInfo<?> outputDomain = (ImmutableOutputInfo<?>) OutputInfo.deserialize(proto.getOutputDomain());
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
        if (!(o instanceof ModelDataCarrier)) {
            return false;
        }
        ModelDataCarrier<?> that = (ModelDataCarrier<?>) o;
        return generatesProbabilities == that.generatesProbabilities && name.equals(that.name)
                && provenance.equals(that.provenance) && featureDomain.equals(that.featureDomain)
                && outputDomain.equals(that.outputDomain) && tribuoVersion.equals(that.tribuoVersion);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, provenance, featureDomain, outputDomain,
                generatesProbabilities, tribuoVersion);
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

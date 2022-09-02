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
import com.oracle.labs.mlrg.olcut.config.protobuf.protos.RootProvenanceProto;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.FeatureMap;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.protos.core.DatasetDataProto;
import org.tribuo.provenance.DataProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Serialization carrier for common fields in Dataset.
 * <p>
 * Likely to be a record one day.
 */
public final class DatasetDataCarrier<T extends Output<T>> {
    private static final ProtoProvenanceSerialization PROVENANCE_SERIALIZER = new ProtoProvenanceSerialization(false);

    /**
     * The data provenance.
     */
    private final DataProvenance provenance;

    /**
     * The features this dataset contains.
     */
    private final FeatureMap featureDomain;

    /**
     * The outputs this dataset contains.
     */
    private final OutputInfo<T> outputDomain;

    /**
     * The transformation provenances.
     */
    private final List<ObjectProvenance> transformProvenances;

    /**
     * The output factory.
     */
    private final OutputFactory<T> outputFactory;

    /**
     * The Tribuo version string.
     */
    private final String tribuoVersion;

    /**
     * Constructs a new DatasetDataCarrier.
     * <p>
     * Will be the canonical constructor for the record form.
     *
     * @param provenance           The data provenance.
     * @param featureDomain        The feature domain.
     * @param outputDomain         The output domain.
     * @param outputFactory        The output factory.
     * @param transformProvenances The transform provenances.
     * @param tribuoVersion        The Tribuo version string.
     */
    public DatasetDataCarrier(DataProvenance provenance, FeatureMap featureDomain, OutputInfo<T> outputDomain, OutputFactory<T> outputFactory, List<ObjectProvenance> transformProvenances, String tribuoVersion) {
        this.provenance = provenance;
        this.featureDomain = featureDomain;
        this.outputDomain = outputDomain;
        this.outputFactory = outputFactory;
        this.transformProvenances = Collections.unmodifiableList(transformProvenances);
        this.tribuoVersion = tribuoVersion;
    }

    /**
     * Deserializes a {@link DatasetDataProto} into a {@link DatasetDataCarrier}.
     *
     * @param proto The proto to deserialize.
     * @return The model data.
     */
    @SuppressWarnings({"unchecked", "rawtypes"}) // guarded by a getClass check
    public static DatasetDataCarrier<?> deserialize(DatasetDataProto proto) {
        DataProvenance provenance = (DataProvenance) ProvenanceUtil.unmarshalProvenance(PROVENANCE_SERIALIZER.deserializeFromProto(proto.getProvenance()));
        FeatureMap featureDomain = FeatureMap.deserialize(proto.getFeatureDomain());
        OutputInfo outputDomain = OutputInfo.deserialize(proto.getOutputDomain());
        OutputFactory outputFactory = OutputFactory.deserialize(proto.getOutputFactory());
        if (outputDomain.getDomain().iterator().next().getClass() != outputFactory.getUnknownOutput().getClass()) {
            throw new IllegalStateException("Invalid protobuf, output domain and output factory use different outputs, output domain " + outputDomain.getClass() + ", output factory " + outputFactory.getClass());
        }
        List<ObjectProvenance> transformProvenances = new ArrayList<>();
        for (RootProvenanceProto p : proto.getTransformProvenanceList()) {
            ObjectProvenance prov = ProvenanceUtil.unmarshalProvenance(PROVENANCE_SERIALIZER.deserializeFromProto(p));
            transformProvenances.add(prov);
        }
        String tribuoVersion = proto.getTribuoVersion();
        return new DatasetDataCarrier<>(provenance, featureDomain, outputDomain, outputFactory, transformProvenances, tribuoVersion);
    }

    /**
     * Serializes this object to a protobuf.
     *
     * @return The protobuf representation.
     */
    public DatasetDataProto serialize() {
        DatasetDataProto.Builder builder = DatasetDataProto.newBuilder();

        builder.setFeatureDomain(featureDomain.serialize());
        builder.setOutputDomain(outputDomain.serialize());
        builder.setOutputFactory(outputFactory.serialize());
        builder.setProvenance(PROVENANCE_SERIALIZER.serializeToProto(ProvenanceUtil.marshalProvenance(provenance)));
        for (ObjectProvenance o : transformProvenances) {
            builder.addTransformProvenance(PROVENANCE_SERIALIZER.serializeToProto(ProvenanceUtil.marshalProvenance(o)));
        }
        builder.setTribuoVersion(tribuoVersion);

        return builder.build();
    }

    /**
     * Get the dataset provenance.
     *
     * @return The dataset provenance.
     */
    public DataProvenance provenance() {
        return provenance;
    }

    /**
     * Get the feature domain.
     *
     * @return The feature domain.
     */
    public FeatureMap featureDomain() {
        return featureDomain;
    }

    /**
     * Get the output domain.
     *
     * @return The output domain.
     */
    public OutputInfo<T> outputDomain() {
        return outputDomain;
    }

    /**
     * Get the transform provenances.
     *
     * @return The transform provenances.
     */
    public List<ObjectProvenance> transformProvenances() {
        return transformProvenances;
    }

    /**
     * Get the output factory.
     *
     * @return The output factory.
     */
    public OutputFactory<T> outputFactory() {
        return outputFactory;
    }

    /**
     * Gets the Tribuo version string.
     *
     * @return The Tribuo version string.
     */
    public String tribuoVersion() {
        return tribuoVersion;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DatasetDataCarrier<?> that = (DatasetDataCarrier<?>) o;
        return provenance.equals(that.provenance) && featureDomain.equals(that.featureDomain) && outputDomain.equals(that.outputDomain) && transformProvenances.equals(that.transformProvenances) && outputFactory.equals(that.outputFactory) && tribuoVersion.equals(that.tribuoVersion);
    }

    @Override
    public int hashCode() {
        return Objects.hash(provenance, featureDomain, outputDomain, transformProvenances, outputFactory, tribuoVersion);
    }

    @Override
    public String toString() {
        return "DatasetDataCarrier{" +
            "provenance=" + provenance +
            ", featureDomain=" + featureDomain +
            ", outputDomain=" + outputDomain +
            ", transformProvenances=" + transformProvenances +
            ", outputFactory=" + outputFactory +
            ", tribuoVersion=" + tribuoVersion +
            '}';
    }
}

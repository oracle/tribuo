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

package org.tribuo;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.FeatureSetProto;
import org.tribuo.protos.core.SelectedFeatureSetProto;
import org.tribuo.provenance.FeatureSetProvenance;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A record-like class for a selected feature set.
 * <p>
 * Uses record style accessors as it may be refactored into a record one day.
 */
@ProtoSerializableClass(serializedDataClass = SelectedFeatureSetProto.class, version = SelectedFeatureSet.CURRENT_VERSION)
public final class SelectedFeatureSet implements ProtoSerializable<FeatureSetProto>, Provenancable<FeatureSetProvenance>, Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @ProtoSerializableField
    private final List<String> featureNames;

    @ProtoSerializableField
    private final List<Double> featureScores;

    @ProtoSerializableField
    private final FeatureSetProvenance provenance;

    @ProtoSerializableField(name="ordered")
    private final boolean isOrdered;

    /**
     * Create a selected feature set.
     * @param featureNames The feature names.
     * @param featureScores The feature scores.
     * @param isOrdered Is this feature set ordered?
     * @param provenance The provenance of the feature selection.
     */
    public SelectedFeatureSet(List<String> featureNames, List<Double> featureScores, boolean isOrdered, FeatureSetProvenance provenance) {
        this.featureNames = Collections.unmodifiableList(featureNames);
        this.featureScores = Collections.unmodifiableList(featureScores);
        this.isOrdered = isOrdered;
        this.provenance = provenance;
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
    public static SelectedFeatureSet deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        SelectedFeatureSetProto proto = message.unpack(SelectedFeatureSetProto.class);
        if (proto.getFeatureNamesCount() != proto.getFeatureScoresCount()) {
            throw new IllegalStateException("Invalid protobuf, mismatch between the number of features and the number of scores, found " + proto.getFeatureNamesCount() + " features and " + proto.getFeatureScoresCount() + " scores");
        }
        ObjectProvenance prov = ProvenanceUtil.unmarshalProvenance(PROVENANCE_SERIALIZER.deserializeFromProto(proto.getProvenance()));
        if (!(prov instanceof FeatureSetProvenance)) {
            throw new IllegalStateException("Invalid protobuf, provenance was not a FeatureSetProvenance, found " + prov.getClass());
        }
        FeatureSetProvenance fsProv = (FeatureSetProvenance) prov;
        return new SelectedFeatureSet(proto.getFeatureNamesList(),proto.getFeatureScoresList(),proto.getOrdered(),fsProv);
    }

    /**
     * The selected feature names in a possibly ordered list.
     * @return The selected feature names.
     */
    public List<String> featureNames() {
        return featureNames;
    }

    /**
     * The selected feature scores in a possibly ordered list.
     * <p>
     * If the algorithm did not produce scores then these values are all {@link Double#NaN}.
     * @return The selected feature scores.
     */
    public List<Double> featureScores() {
        return featureScores;
    }

    /**
     * The provenance of the feature set.
     * @return The feature set provenance.
     */
    public FeatureSetProvenance provenance() {
        return provenance;
    }

    @Override
    public FeatureSetProvenance getProvenance() {
        return provenance;
    }

    /**
     * Is this feature set ordered?
     * @return True if the set is ordered.
     */
    public boolean isOrdered() {
        return isOrdered;
    }

    @Override
    public FeatureSetProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Checks if this {@code SelectedFeatureSet} is equal to the supplied object.
     * <p>
     * Equals is defined as containing the same features, in the same order, with the same scores, and with
     * the same provenance information. As that provenance includes machine information and timestamps, this
     * means equals is defined as did this object derive from the same computation as the supplied object.
     * @param o The object to test.
     * @return True if they are equal.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SelectedFeatureSet that = (SelectedFeatureSet) o;
        return isOrdered == that.isOrdered && featureNames.equals(that.featureNames) && featureScores.equals(that.featureScores) && provenance.equals(that.provenance);
    }

    /**
     * Computes the hash code.
     * <p>
     * The hash code depends on the provenance object, and is not just a function of the feature names and scores, to
     * be consistent with the definition of equals.
     * @return The hash code.
     */
    @Override
    public int hashCode() {
        return Objects.hash(featureNames, featureScores, provenance, isOrdered);
    }

    @Override
    public String toString() {
        return "SelectedFeatureSet{" +
            "featureNames=" + featureNames +
            ", featureScores=" + featureScores +
            ", isOrdered=" + isOrdered +
            '}';
    }
}

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

package org.tribuo.clustering;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.clustering.protos.ClusterIDProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputProto;

import java.util.Objects;

/**
 * A clustering id.
 * <p>
 * The id is an int, referring to a cluster stored in the model.
 * Optionally contains a score representing the strength of association
 * with that cluster, if available.
 * <p>
 * The id is {@link ClusterID#UNASSIGNED} if the output is not assigned to a
 * cluster (e.g., before the {@link Model} has been trained).
 */
@ProtoSerializableClass(serializedDataClass = ClusterIDProto.class, version = 0)
public class ClusterID implements Output<ClusterID> {
    private static final long serialVersionUID = 1L;

    /**
     * The sentinel unassigned cluster ID.
     */
    public static final int UNASSIGNED = -1;

    @ProtoSerializableField
    private final int id;

    @ProtoSerializableField
    private final double score;

    /**
     * Creates a ClusterID with the sentinel score of {@link Double#NaN}.
     * @param id The cluster id number.
     */
    public ClusterID(int id) {
        this(id,Double.NaN);
    }

    /**
     * Creates a ClusterID with the specified id number and score.
     * @param id The cluster id number.
     * @param score The score.
     */
    public ClusterID(int id, double score) {
        this.id = id;
        this.score = score;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ClusterID deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        ClusterIDProto proto = message.unpack(ClusterIDProto.class);
        ClusterID lbl = new ClusterID(proto.getId(),proto.getScore());
        return lbl;
    }

    @Override
    public OutputProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Get a real valued score for this ClusterID.
     * <p>
     * If the score is not set then it returns Double.NaN.
     * @return The predicted score for this cluster id.
     */
    public double getScore() {
        return score;
    }

    /**
     * Gets the cluster id number.
     * @return A int.
     */
    public int getID() {
        return id;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ClusterID)) return false;
        ClusterID that = (ClusterID) o;
        return id == that.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public boolean fullEquals(ClusterID o) {
        if (this == o) return true;

        if ((!(Double.isNaN(o.score) && Double.isNaN(score))) && (Double.compare(o.score, score) != 0)) return false;
        return id == o.id;
    }

    @Override
    public String toString() {
        if (Double.isNaN(score)) {
            return ""+id;
        } else {
            return "("+id+","+score+")";
        }
    }

    @Override
    public ClusterID copy() {
        return new ClusterID(id, score);
    }

    /**
     * Returns "id" or "id,score=idScore".
     * @param includeConfidence Include whatever confidence score the clusterID contains, if known.
     * @return A string representing this ClusterID, suitable for csv or json serialization.
     */
    @Override
    public String getSerializableForm(boolean includeConfidence) {
        if (includeConfidence && !Double.isNaN(score)) {
            return id + ",score=" + score;
        } else {
            return ""+id;
        }
    }
}

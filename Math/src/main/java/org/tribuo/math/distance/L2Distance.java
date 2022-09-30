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

package org.tribuo.math.distance;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.protos.DistanceProto;

/**
 * L2 (or Euclidean) distance.
 */
public final class L2Distance implements Distance {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs an L2 distance function.
     */
    public L2Distance() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     */
    public static L2Distance deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new L2Distance();
    }

    @Override
    public DistanceProto serialize() {
        DistanceProto.Builder kernelProto = DistanceProto.newBuilder();
        kernelProto.setClassName(this.getClass().getName());
        kernelProto.setVersion(CURRENT_VERSION);
        return kernelProto.build();
    }

    @Override
    public double computeDistance(SGDVector first, SGDVector second) {
        return first.l2Distance(second);
    }

    @Override
    public String toString() {
        return "L2Distance()";
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof L2Distance;
    }

    @Override
    public int hashCode() {
        return 31;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Distance");
    }
}

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

package org.tribuo.math.kernel;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.protos.KernelProto;
import org.tribuo.math.protos.SigmoidKernelProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.Objects;

/**
 * A sigmoid kernel, tanh(gamma*u.dot(v) + intercept).
 */
@ProtoSerializableClass(version = Sigmoid.CURRENT_VERSION, serializedDataClass = SigmoidKernelProto.class)
public class Sigmoid implements Kernel {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @Config(mandatory = true,description="Coefficient to multiply the dot product by.")
    @ProtoSerializableField
    private double gamma;

    @Config(mandatory = true,description="Scalar intercept to add to the dot product.")
    @ProtoSerializableField
    private double intercept;

    /**
     * For olcut.
     */
    private Sigmoid() {}

    /**
     * A sigmoid kernel, tanh(gamma*u.dot(v) + intercept).
     * @param gamma A scalar coefficient.
     * @param intercept An additive coefficient.
     */
    public Sigmoid(double gamma, double intercept) {
        this.gamma = gamma;
        this.intercept = intercept;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static Sigmoid deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        SigmoidKernelProto kernelProto = message.unpack(SigmoidKernelProto.class);
        return new Sigmoid(kernelProto.getGamma(),kernelProto.getIntercept());
    }

    @Override
    public KernelProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public double similarity(SparseVector a, SparseVector b) {
        return Math.tanh(gamma * a.dot(b) + intercept);
    }

    @Override
    public String toString() {
        return "Sigmoid(gamma="+gamma+",intercept="+intercept+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"Kernel");
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Sigmoid sigmoid = (Sigmoid) o;
        return Double.compare(sigmoid.gamma, gamma) == 0 && Double.compare(sigmoid.intercept, intercept) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(gamma, intercept);
    }
}

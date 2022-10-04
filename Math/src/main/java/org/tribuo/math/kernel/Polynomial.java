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
import org.tribuo.math.protos.PolynomialKernelProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.Objects;

/**
 * A polynomial kernel, (gamma*u.dot(v) + intercept)^degree.
 */
@ProtoSerializableClass(version = Polynomial.CURRENT_VERSION, serializedDataClass = PolynomialKernelProto.class)
public class Polynomial implements Kernel {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @Config(mandatory = true,description="Coefficient to multiply the dot product by.")
    @ProtoSerializableField
    private double gamma;

    @Config(mandatory = true,description="Scalar to add to the dot product.")
    @ProtoSerializableField
    private double intercept;

    @Config(mandatory = true,description="Degree of the polynomial.")
    @ProtoSerializableField
    private double degree;

    /**
     * For olcut.
     */
    private Polynomial() {}

    /**
     * A polynomial kernel, (gamma*u.dot(v) + intercept)^degree.
     * @param gamma The scalar coefficient.
     * @param intercept An additive coefficient.
     * @param degree The degree of the polynomial.
     */
    public Polynomial(double gamma, double intercept, double degree) {
        this.gamma = gamma;
        this.intercept = intercept;
        this.degree = degree;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static Polynomial deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        PolynomialKernelProto kernelProto = message.unpack(PolynomialKernelProto.class);
        return new Polynomial(kernelProto.getGamma(),kernelProto.getIntercept(),kernelProto.getDegree());
    }

    @Override
    public KernelProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public double similarity(SparseVector a, SparseVector b) {
        return Math.pow(gamma * a.dot(b) + intercept, degree);
    }

    @Override
    public String toString() {
        return "Polynomial(gamma="+gamma+",intercept="+intercept+",degree="+degree+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"Kernel");
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Polynomial that = (Polynomial) o;
        return Double.compare(that.gamma, gamma) == 0 && Double.compare(that.intercept, intercept) == 0 && Double.compare(that.degree, degree) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(gamma, intercept, degree);
    }
}

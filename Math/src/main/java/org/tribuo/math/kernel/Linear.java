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
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.protos.KernelProto;

/**
 * A linear kernel, u.dot(v).
 */
public class Linear implements Kernel {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * A linear kernel, u.dot(v).
     */
    public Linear() { }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static Linear deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new Linear();
    }

    @Override
    public KernelProto serialize() {
        KernelProto.Builder kernelProto = KernelProto.newBuilder();
        kernelProto.setClassName(this.getClass().getName());
        kernelProto.setVersion(CURRENT_VERSION);
        return kernelProto.build();
    }

    @Override
    public double similarity(SparseVector a, SparseVector b) {
        return a.dot(b);
    }

    @Override
    public String toString() {
        return "Linear()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"Kernel");
    }

    @Override
    public boolean equals(Object o) {
        return o.getClass().equals(Linear.class);
    }

    @Override
    public int hashCode() {
        return 31;
    }
}

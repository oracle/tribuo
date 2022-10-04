/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.hash;

import com.google.protobuf.Any;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.protos.core.HasherProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Hashes names using String.hashCode().
 * <p>
 * HashCodeHasher does not serialize the salt in its serialized forms, and
 * thus the salt must be set after deserialization.
 */
@ProtoSerializableClass(version = HashCodeHasher.CURRENT_VERSION)
public final class HashCodeHasher extends Hasher {
    private static final long serialVersionUID = 2L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @Config(mandatory = true, redact = true, description="Salt used in the hash.")
    private transient String salt = null;

    private static final HashCodeHasherProvenance provenance = new HashCodeHasherProvenance();

    /**
     * for olcut.
     */
    private HashCodeHasher() { }

    /**
     * Constructs a HashCodeHasher using the specified salt value.
     * @param salt The salt value.
     */
    public HashCodeHasher(String salt) {
        this.salt = salt;
        postConfig();
    }

    /**
     * Deserialization factory.
     * <p>
     * Note the salt must be set after the hasher has been deserialized.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static HashCodeHasher deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        return new HashCodeHasher();
    }

    @Override
    public HasherProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public String hash(String name) {
        if (salt == null) {
            throw new IllegalStateException("Salt not set");
        }
        String salted = salt + name;
        return ""+salted.hashCode();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws PropertyException{
        if (salt == null) {
            throw new PropertyException("","salt","Salt not set in HashCodeHasher.");
        } else if (!Hasher.validateSalt(salt)) {
            throw new PropertyException("","salt","Salt does not meet the requirements for a salt.");
        }
    }

    @Override
    public void setSalt(String salt) {
        if (Hasher.validateSalt(salt)) {
            this.salt = salt;
        } else {
            throw new IllegalArgumentException("Salt: '" + salt + ", does not meet the requirements for a salt.");
        }
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return provenance;
    }

    @Override
    public String toString() {
        return "HashCodeHasher()";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        HashCodeHasher that = (HashCodeHasher) o;
        return Objects.equals(salt, that.salt);
    }

    @Override
    public int hashCode() {
        return Objects.hash(salt);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        salt = null;
    }

    /**
     * Provenance for the {@link HashCodeHasher}.
     */
    public final static class HashCodeHasherProvenance implements ConfiguredObjectProvenance {
        private static final long serialVersionUID = 1L;

        HashCodeHasherProvenance() {}

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public HashCodeHasherProvenance(Map<String, Provenance> map) { }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            return Collections.singletonMap("salt",new StringProvenance("salt",""));
        }

        @Override
        public String getClassName() {
            return HashCodeHasher.class.getName();
        }

        @Override
        public String toString() {
            return generateString("Hasher");
        }

        @Override
        public int hashCode() {
            return 31;
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof HashCodeHasher;
        }
    }
}

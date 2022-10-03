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
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.HasherProto;
import org.tribuo.protos.core.ModHashCodeHasherProto;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Hashes names using String.hashCode(), then reduces the dimension.
 * <p>
 * ModHashCodeHasher does not serialize the salt in its serialized forms, and
 * thus the salt must be set after deserialization.
 */
@ProtoSerializableClass(version = ModHashCodeHasher.CURRENT_VERSION, serializedDataClass = ModHashCodeHasherProto.class)
public final class ModHashCodeHasher extends Hasher {
    private static final long serialVersionUID = 2L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    static final String DIMENSION = "dimension";

    @Config(mandatory = true,redact = true,description="Salt used in the hash.")
    private transient String salt = null;

    @ProtoSerializableField
    @Config(mandatory = true,description="Range of the hashing function.")
    private int dimension = 100;

    private ModHashCodeHasherProvenance provenance;

    /**
     * for olcut.
     */
    private ModHashCodeHasher() { }

    /**
     * Constructs a ModHashCodeHasher with a fixed dimensionality of 100.
     * @param salt The salt value.
     */
    public ModHashCodeHasher(String salt) {
        this(100,salt);
    }

    /**
     * Constructs a ModHashCodeHasher with the supplied parameters.
     * @param dimension The dimensionality.
     * @param salt The salt value.
     */
    public ModHashCodeHasher(int dimension, String salt) {
        this.dimension = dimension;
        this.salt = salt;
        postConfig();
    }

    /**
     * Deserialization constructor.
     * <p>
     * Note the salt must be set after the hasher has been deserialized.
     * @param version The version number.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ModHashCodeHasher deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ModHashCodeHasherProto proto = message.unpack(ModHashCodeHasherProto.class);
        ModHashCodeHasher obj = new ModHashCodeHasher();
        obj.dimension = proto.getDimension();
        obj.provenance = new ModHashCodeHasherProvenance(obj.dimension);
        return obj;
    }

    @Override
    public HasherProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws PropertyException {
        if (salt == null) {
            throw new PropertyException("","salt","Salt not set in ModHashCodeHasher.");
        } else if (!Hasher.validateSalt(salt)) {
            throw new PropertyException("","salt","Salt does not meet the requirements for a salt.");
        }
        this.provenance = new ModHashCodeHasherProvenance(dimension);
    }

    @Override
    public String hash(String name) {
        if (salt == null) {
            throw new IllegalStateException("Salt not set");
        }
        String salted = salt + name;
        return ""+(salted.hashCode() % dimension);
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
        return "ModHashCodeHasher(dimension=" + dimension + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        ModHashCodeHasher that = (ModHashCodeHasher) o;
        return dimension == that.dimension && Objects.equals(salt, that.salt);
    }

    @Override
    public int hashCode() {
        return Objects.hash(salt, dimension);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        salt = null;
    }

    /**
     * Provenance for the {@link ModHashCodeHasher}.
     */
    public final static class ModHashCodeHasherProvenance implements ConfiguredObjectProvenance {
        private static final long serialVersionUID = 1L;

        private final IntProvenance dimension;

        ModHashCodeHasherProvenance(int dimension) {
            this.dimension = new IntProvenance(DIMENSION,dimension);
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public ModHashCodeHasherProvenance(Map<String, Provenance> map) {
            dimension = ObjectProvenance.checkAndExtractProvenance(map,DIMENSION,IntProvenance.class,ModHashCodeHasherProvenance.class.getSimpleName());
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            Map<String,Provenance> map = new HashMap<>();
            map.put("saltStr",new StringProvenance("saltStr",""));
            map.put(DIMENSION,dimension);
            return map;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof ModHashCodeHasherProvenance)) return false;
            ModHashCodeHasherProvenance pairs = (ModHashCodeHasherProvenance) o;
            return dimension.equals(pairs.dimension);
        }

        @Override
        public int hashCode() {
            return Objects.hash(dimension);
        }

        @Override
        public String getClassName() {
            return ModHashCodeHasher.class.getName();
        }

        @Override
        public String toString() {
            return generateString("Hasher");
        }
    }
}

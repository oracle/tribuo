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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Hashes names using String.hashCode(), then reduces the dimension.
 */
public final class ModHashCodeHasher extends Hasher {
    private static final long serialVersionUID = 2L;

    static final String DIMENSION = "dimension";

    @Config(mandatory = true,redact = true,description="Salt used in the hash.")
    private transient String salt = null;

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

/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.Map;

/**
 * Hashes names using String.hashCode().
 */
public final class HashCodeHasher extends Hasher {
    private static final long serialVersionUID = 2L;

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
    }

    @Override
    public String hash(String name) {
        if (salt == null) {
            throw new IllegalStateException("Salt not set");
        }
        String salted = salt + name;
        return ""+salted.hashCode();
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

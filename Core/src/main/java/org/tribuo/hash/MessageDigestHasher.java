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
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Hashes Strings using the supplied MessageDigest type.
 */
public final class MessageDigestHasher extends Hasher {
    private static final long serialVersionUID = 3L;

    public static final Charset utf8Charset = StandardCharsets.UTF_8;

    static final String HASH_TYPE = "hashType";

    @Config(mandatory = true,description="MessageDigest hashing function.")
    private String hashType;

    private transient ThreadLocal<MessageDigest> md;

    /**
     * Only used by olcut.
     */
    @Config(mandatory = true,description="Salt used in the hash.",redact=true)
    private transient String saltStr = null;

    private transient byte[] salt = null;

    private MessageDigestHasherProvenance provenance;

    /**
     * For olcut.
     */
    private MessageDigestHasher() {}

    public MessageDigestHasher(String hashType, String salt) {
        this.hashType = hashType;
        this.salt = salt.getBytes(utf8Charset);
        this.md = ThreadLocal.withInitial(getDigestSupplier(hashType));
        MessageDigest d = this.md.get(); // To trigger the unsupported digest exception.
        this.provenance = new MessageDigestHasherProvenance(hashType);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws PropertyException {
        if (saltStr != null) {
            salt = saltStr.getBytes(utf8Charset);
        } else {
            throw new PropertyException("","saltStr","Salt not set in MessageDigestHasher.");
        }
        md = ThreadLocal.withInitial(getDigestSupplier(hashType));
        try {
            MessageDigest d = md.get();// To trigger the unsupported digest exception.
        } catch (IllegalArgumentException e) {
            throw new PropertyException("","hashType","Unsupported hashType = " + hashType);
        }
        this.provenance = new MessageDigestHasherProvenance(hashType);
    }

    @Override
    public String hash(String input) {
        if (salt == null) {
            throw new IllegalStateException("Salt not set.");
        }
        MessageDigest localDigest = md.get();
        localDigest.reset();
        localDigest.update(salt);
        byte[] hash = localDigest.digest(input.getBytes(utf8Charset));
        return Base64.getEncoder().encodeToString(hash);
    }

    @Override
    public void setSalt(String salt) {
        if (Hasher.validateSalt(salt)) {
            this.salt = salt.getBytes(utf8Charset);
        } else {
            throw new IllegalArgumentException("Salt: '" + salt + "', does not meet the requirements for a salt.");
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        salt = null;
        saltStr = null;
        md = ThreadLocal.withInitial(getDigestSupplier(hashType));
    }

    @Override
    public String toString() {
        return "MessageDigestHasher(algorithm="+md.get().getAlgorithm()+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return provenance;
    }

    /**
     * Creates a supplier for the specified hash type.
     * @param hashType The hash type, used to specify the MessageDigest implementation.
     * @return A supplier for the MessageDigest.
     */
    public static Supplier<MessageDigest> getDigestSupplier(String hashType) {
        return () -> { try { return MessageDigest.getInstance(hashType); } catch (NoSuchAlgorithmException e) { throw new IllegalArgumentException("Unsupported hashType = " + hashType,e);}};
    }

    /**
     * Provenance for {@link MessageDigestHasher}.
     */
    public final static class MessageDigestHasherProvenance implements ConfiguredObjectProvenance {
        private static final long serialVersionUID = 1L;

        private final StringProvenance hashType;

        MessageDigestHasherProvenance(String hashType) {
            this.hashType = new StringProvenance(HASH_TYPE,hashType);
        }

        public MessageDigestHasherProvenance(Map<String, Provenance> map) {
            hashType = ObjectProvenance.checkAndExtractProvenance(map,HASH_TYPE,StringProvenance.class,MessageDigestHasherProvenance.class.getSimpleName());
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            Map<String,Provenance> map = new HashMap<>();
            map.put("saltStr",new StringProvenance("saltStr",""));
            map.put(HASH_TYPE,hashType);
            return map;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof MessageDigestHasherProvenance)) return false;
            MessageDigestHasherProvenance pairs = (MessageDigestHasherProvenance) o;
            return hashType.equals(pairs.hashType);
        }

        @Override
        public int hashCode() {
            return Objects.hash(hashType);
        }

        @Override
        public String getClassName() {
            return MessageDigestHasher.class.getName();
        }

        @Override
        public String toString() {
            return generateString("Hasher");
        }
    }

}

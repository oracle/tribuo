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

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.core.HasherProto;

import java.io.Serializable;

/**
 * An abstract base class for hash functions used to hash the names of features.
 */
public abstract class Hasher implements Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable,
        ProtoSerializable<HasherProto>  {
    private static final long serialVersionUID = 2L;

    /**
     * The minimum length of the salt. Salts shorter than this will not validate.
     */
    public static final int MIN_LENGTH = 8;

    /**
     * Hashes the supplied input using the hashing function.
     * @param input The input to hash.
     * @return A String representation of the hashed output.
     */
    public abstract String hash(String input);

    /**
     * The salt is transient, it must be set **to the same value as it was trained with**
     * after the {@link org.tribuo.Model} is deserialized.
     * @param salt Salt value.
     */
    public abstract void setSalt(String salt);

    /**
     * Salt validation is currently a test to see if the string is longer than {@link Hasher#MIN_LENGTH}.
     * <p>
     * When this method is updated Hasher must update it's serialVersionUID, this ensures that
     * serialised instances of downstream classes which call this method are invalidated, as
     * changes to validateSalt may invalidate old salts, and there is no other way
     * to communicate this to the developer.
     * @param salt String to validate.
     * @return True if the salt is valid, false otherwise.
     */
    public static boolean validateSalt(String salt) {
        return salt.length() > MIN_LENGTH;
    }

}

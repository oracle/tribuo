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

package org.tribuo.hash;

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class HasherTests {

    private static final String VALID_SALT = "aSaltString";
    private static final String ANOTHER_VALID_SALT = "anotherSaltString";
    private static final String INVALID_SALT = "tooShort";
    private static final String ANOTHER_INVALID_SALT = "wayShort";

    private static void hasherSimpleTest(Hasher hasher, String firstHashedStr, String secondHashedStr) {
        assertEquals(firstHashedStr, hasher.hash("AStringToHash"));
        hasher.setSalt(ANOTHER_VALID_SALT);
        assertEquals(secondHashedStr, hasher.hash("AStringToHash"));
    }

    @Test
    public void hashCodeHasherSimpleTest() {
        Hasher hasher = new HashCodeHasher(VALID_SALT);
        hasherSimpleTest(hasher, "365284915", "509047121");
    }

    @Test
    public void hashCodeHasherConstructThrows() {
        assertThrows(PropertyException.class, () -> new HashCodeHasher(INVALID_SALT));
    }

    @Test
    public void hashCodeHasherSetSaltThrows() {
        Hasher hasher = new HashCodeHasher(VALID_SALT);
        assertThrows(IllegalArgumentException.class, () -> hasher.setSalt(ANOTHER_INVALID_SALT));
    }

    @Test void messageDigestHasherSimpleTest() {
        Hasher hasher = new MessageDigestHasher("SHA-256", VALID_SALT);
        hasherSimpleTest(hasher, "akemHZ4RWFogeccq7nVBoIYDeqES2oEb2yKamVIOz+k=",
            "pFa7A2fVQHsumKMVkgoLnagSxzrHwVgBCAOvq52c9XI=");
    }

    @Test
    public void messageDigestHasherConstructThrows() {
        assertThrows(IllegalArgumentException.class, () -> new MessageDigestHasher("SHA-256", INVALID_SALT));
    }

    @Test
    public void messageDigestHasherSetSaltThrows() {
        Hasher hasher = new MessageDigestHasher("SHA1", VALID_SALT);
        assertThrows(IllegalArgumentException.class, () -> hasher.setSalt(ANOTHER_INVALID_SALT));
    }

    @Test
    public void modHashCodeHasherSimpleTest() {
        Hasher hasher = new ModHashCodeHasher(VALID_SALT);
        hasherSimpleTest(hasher, "15", "21");
    }

    @Test
    public void modHashCodeHasherConstructThrows() {
        assertThrows(PropertyException.class, () -> new ModHashCodeHasher(INVALID_SALT));
    }

    @Test
    public void modHashCodeHasherSetSaltThrows() {
        Hasher hasher = new ModHashCodeHasher(VALID_SALT);
        assertThrows(IllegalArgumentException.class, () -> hasher.setSalt(ANOTHER_INVALID_SALT));
    }
}

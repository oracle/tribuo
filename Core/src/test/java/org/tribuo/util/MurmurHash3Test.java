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

package org.tribuo.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link MurmurHash3}.
 */
class MurmurHash3Test {

    @Test
    void testFmix32() {
        assertEquals(1364076727, MurmurHash3.fmix32(1));
        assertEquals(-2114883783, MurmurHash3.fmix32(-1));
        assertEquals(-383449968, MurmurHash3.fmix32(10));
        assertEquals(-36807446, MurmurHash3.fmix32(100));
        assertEquals(-1434176238, MurmurHash3.fmix32(21323));
    }

    @Test
    void testFmix64() {
        assertEquals(-5451962507482445012L, MurmurHash3.fmix64(1));
        assertEquals(7256831767414464289L, MurmurHash3.fmix64(-1));
        assertEquals(7233188113542599437L, MurmurHash3.fmix64(10));
        assertEquals(-1819968182471218078L, MurmurHash3.fmix64(100));
        assertEquals(3402563454534931576L, MurmurHash3.fmix64(21323));
    }

}
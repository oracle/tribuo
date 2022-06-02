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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import org.junit.jupiter.api.Test;
import org.tribuo.hash.HashCodeHasher;
import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.hash.Hasher;
import org.tribuo.protos.core.FeatureDomainProto;
import org.tribuo.protos.core.HasherProto;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class FeatureMapTest {

    public static FeatureMap buildMap() {
        SplittableRandom rng = new SplittableRandom(12345);
        MutableFeatureMap fm = new MutableFeatureMap();

        // Add categorical features
        for (int i = 0; i < 100; i++) {
            fm.add("A", rng.nextInt(10));
            fm.add("B", rng.nextInt(2));
            fm.add("C", rng.nextInt(40));
        }
        for (int i = 0; i < 50; i++) {
            fm.add("D", rng.nextInt(25));
            fm.add("E", rng.nextInt(1));
        }

        // Add real features
        for (int i = 0; i < 100; i++) {
            fm.add("F", rng.nextInt(100));
            fm.add("G", rng.nextDouble(2));
            fm.add("H", rng.nextDouble());
        }

        return fm;
    }

    @Test
    public void testBasic() {
        FeatureMap fm = buildMap();
        FeatureMap otherFm = buildMap();

        assertEquals(fm, otherFm);

        int catCount = 0;
        int realCount = 0;
        for (VariableInfo i : fm) {
            if (i instanceof CategoricalInfo) {
                catCount++;
            } else if (i instanceof RealInfo) {
                realCount++;
            }
        }

        assertEquals(5,catCount);
        assertEquals(3,realCount);
        assertEquals(8,fm.size());

        ImmutableFeatureMap ifm = new ImmutableFeatureMap(fm);

        assertNotEquals(fm,ifm);
        assertEquals(8,ifm.size());
    }

    @Test
    public void testSerialization() {
        FeatureMap fm = buildMap();
        FeatureDomainProto fmProto = fm.serialize();
        FeatureMap deserFm = FeatureMap.deserialize(fmProto);
        assertEquals(fm, deserFm);

        ImmutableFeatureMap ifm = new ImmutableFeatureMap(fm);
        FeatureDomainProto ifmProto = ifm.serialize();
        FeatureMap deserIfm = FeatureMap.deserialize(ifmProto);
        assertEquals(ifm, deserIfm);
    }

    @Test
    public void testHashed() {
        FeatureMap fm = buildMap();
        String salt = "This is a salt";
        Hasher hasher = new HashCodeHasher(salt);
        HashedFeatureMap hfm = HashedFeatureMap.generateHashedFeatureMap(fm,hasher);

        // Note this check is only true if the hashing didn't induce collisions.
        assertEquals(fm.size(),hfm.size);

        FeatureDomainProto hfmProto = hfm.serialize();
        FeatureMap deserHfm = FeatureMap.deserialize(hfmProto);
        // Serialization intentionally doesn't preserve the salt.
        ((HashedFeatureMap) deserHfm).setSalt(salt);
        assertEquals(hfm, deserHfm);
    }

}

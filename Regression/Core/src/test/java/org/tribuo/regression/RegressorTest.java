/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class RegressorTest {

    /**
     * Tests that the id assingment uses lexicographic ordering unless overridden.
     */
    @Test
    public void testIdAssignment() {
        RegressionFactory factory = new RegressionFactory();

        MutableRegressionInfo info = (MutableRegressionInfo) factory.generateInfo();

        String[] dimNames = new String[]{
                "A","B","C","dim0","dim1","dim2","dim3","dim4","dim5"
        };

        double[] dimValues = new double[]{
                0,1,2,3,4,5,6,7,8
        };

        Regressor r = new Regressor(dimNames,dimValues);

        info.observe(r);

        ImmutableRegressionInfo immutableInfo = (ImmutableRegressionInfo) info.generateImmutableOutputInfo();

        for (Pair<Integer,Regressor> p : immutableInfo) {
            assertTrue(p.getB() instanceof Regressor.DimensionTuple);
            assertEquals(dimNames[p.getA()],((Regressor.DimensionTuple)p.getB()).getName());
        }

        assertTrue(immutableInfo.validateMapping());

        // Check mapping functions are the identity
        int[] actualIDtoNative = immutableInfo.getIDtoNaturalOrderMapping();
        int[] trueOrdering = new int[]{0,1,2,3,4,5,6,7,8};
        assertArrayEquals(trueOrdering,actualIDtoNative);
        int[] actualNativeToID = immutableInfo.getNaturalOrderToIDMapping();
        assertArrayEquals(trueOrdering,actualNativeToID);

        Map<Regressor,Integer> mapping = new HashMap<>();
        mapping.put(new Regressor.DimensionTuple(dimNames[0],dimValues[0]),8);
        mapping.put(new Regressor.DimensionTuple(dimNames[1],dimValues[1]),0);
        mapping.put(new Regressor.DimensionTuple(dimNames[2],dimValues[2]),7);
        mapping.put(new Regressor.DimensionTuple(dimNames[3],dimValues[3]),1);
        mapping.put(new Regressor.DimensionTuple(dimNames[4],dimValues[4]),6);
        mapping.put(new Regressor.DimensionTuple(dimNames[5],dimValues[5]),2);
        mapping.put(new Regressor.DimensionTuple(dimNames[6],dimValues[6]),5);
        mapping.put(new Regressor.DimensionTuple(dimNames[7],dimValues[7]),3);
        mapping.put(new Regressor.DimensionTuple(dimNames[8],dimValues[8]),4);

        ImmutableRegressionInfo mappedInfo = new ImmutableRegressionInfo(info,mapping);

        assertFalse(mappedInfo.validateMapping());

        // Check mapping functions respect the mapping
        int[] mappingIDtoNative = mappedInfo.getIDtoNaturalOrderMapping();
        int[] trueOrderingID = new int[]{1,3,5,7,8,6,4,2,0};
        assertArrayEquals(trueOrderingID,mappingIDtoNative);
        int[] mappingNativeToID = mappedInfo.getNaturalOrderToIDMapping();
        int[] trueOrderingNative = new int[]{8,0,7,1,6,2,5,3,4};
        assertArrayEquals(trueOrderingNative,mappingNativeToID);
    }

    @Test
    public void getsCorrectSerializableForm() {
        Regressor mr = new Regressor(
                new String[]{"a", "b", "c"},
                new double[]{1d, 2d, 3d}
        );
        assertEquals("a=1.0,b=2.0,c=3.0", mr.getSerializableForm(false));
        // Should be the same for includeConfidence either way, since we ignore NaN variances
        assertEquals("a=1.0,b=2.0,c=3.0", mr.getSerializableForm(true));

        Regressor scored = new Regressor(
                new String[]{"a", "b", "c"},
                new double[]{1d, 2d, 3d},
                new double[]{0d, 0d, 0.5}
        );
        assertEquals("a=1.0,b=2.0,c=3.0", scored.getSerializableForm(false));
        assertEquals("a=1.0\u00B10.0,b=2.0\u00B10.0,c=3.0\u00B10.5", scored.getSerializableForm(true));
    }

}
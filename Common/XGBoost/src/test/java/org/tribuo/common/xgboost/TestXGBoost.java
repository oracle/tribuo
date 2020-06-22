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

package org.tribuo.common.xgboost;

import org.tribuo.CategoricalIDInfo;
import org.tribuo.CategoricalInfo;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.impl.ListExample;
import org.tribuo.test.MockOutput;
import org.tribuo.util.Util;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Tests for common XGBoost functionality (example conversion).
 */
public class TestXGBoost {
    @Test
    public void duplicateFeatureIDs() {
        ImmutableFeatureMap fmap = new TestMap();

        ArrayList<Float> data = new ArrayList<>();
        ArrayList<Integer> indices = new ArrayList<>();
        ArrayList<Long> header = new ArrayList<>();
        Example<MockOutput> collision = generateExample(new String[]{"FOO","BAR","BAZ","QUUX"},new double[]{1.0,2.2,3.3,4.4});
        int[] testCollisionIndices = new int[]{0,1,2};
        float[] testCollisionValues = new float[]{4.3f,2.2f,4.4f};
        XGBoostTrainer.convertSingleExample(collision,fmap,data,indices,header,0);
        assertArrayEquals(testCollisionIndices, Util.toPrimitiveInt(indices));
        assertArrayEquals(testCollisionValues, Util.toPrimitiveFloat(data),1e-10f);

        data.clear();
        indices.clear();
        header.clear();
        Example<MockOutput> fakecollision = generateExample(new String[]{"BAR","BAZ","QUUX"},new double[]{2.2,3.3,4.4});
        XGBoostTrainer.convertSingleExample(fakecollision,fmap,data,indices,header,0);
        int[] testFakeCollisionIndices = new int[]{0,1,2};
        float[] testFakeCollisionValues = new float[]{3.3f,2.2f,4.4f};
        assertArrayEquals(testFakeCollisionIndices,Util.toPrimitiveInt(indices));
        assertArrayEquals(testFakeCollisionValues,Util.toPrimitiveFloat(data),1e-10f);
    }

    private static Example<MockOutput> generateExample(String[] names, double[] values) {
        Example<MockOutput> e = new ListExample<>(new MockOutput("MONKEYS"));
        for (int i = 0; i < names.length; i++) {
            e.add(new Feature(names[i],values[i]));
        }
        return e;
    }

    private static class TestMap extends ImmutableFeatureMap {
        private static final long serialVersionUID = 1L;
        public TestMap() {
            super();
            CategoricalIDInfo foo = (new CategoricalInfo("FOO")).makeIDInfo(0);
            m.put("FOO",foo);
            idMap.put(0,foo);
            CategoricalIDInfo bar = (new CategoricalInfo("BAR")).makeIDInfo(1);
            m.put("BAR",bar);
            idMap.put(1,bar);
            CategoricalIDInfo baz = (new CategoricalInfo("BAZ")).makeIDInfo(0);
            m.put("BAZ",baz);
            idMap.put(0,baz);
            CategoricalIDInfo quux = (new CategoricalInfo("QUUX")).makeIDInfo(2);
            m.put("QUUX",quux);
            idMap.put(2,quux);
            size = idMap.size();
        }
    }

}

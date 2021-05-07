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

package org.tribuo.interop.tensorflow;

import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.types.TFloat32;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.MutableFeatureMap;
import org.tribuo.impl.ArrayExample;
import org.tribuo.test.MockOutput;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Basic tests for the image feature transformer.
 */
public class ImageConverterTest {

    private static ImmutableFeatureMap constructFeatureMap() {
        MutableFeatureMap fMap = new MutableFeatureMap();

        fMap.add("A",1);
        fMap.add("B",1);
        fMap.add("C",1);
        fMap.add("D",1);
        fMap.add("E",1);
        fMap.add("F",1);
        fMap.add("G",1);
        fMap.add("H",1);
        fMap.add("I",1);
        fMap.add("J",1);
        fMap.add("K",1);
        fMap.add("L",1);
        fMap.add("M",1);
        fMap.add("N",1);
        fMap.add("O",1);
        fMap.add("P",1);
        fMap.add("Q",1);
        fMap.add("R",1);

        return new ImmutableFeatureMap(fMap);
    }

    private static Example<MockOutput> constructExample() {
        String[] featureNames = new String[]{"A","B","C","D","F","G","H","I","J","K","L","M","O","P","Q","R"};
        double[] featureValues = new double[]{0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17};

        ArrayExample<MockOutput> e = new ArrayExample<>(new MockOutput("Test"),featureNames,featureValues);

        return e;
    }

    @Test
    public void testImageConverter() {
        ImmutableFeatureMap fmap = constructFeatureMap();
        Example<MockOutput> e = constructExample();

        // 3,3,2
        ImageConverter first = new ImageConverter("test",3,3,2);
        FloatNdArray ndarray = (TFloat32) first.convert(e,fmap).getMap().get("test");
        assertEquals( 0, ndarray.getFloat(0,0,0,0), 1e-10);
        assertEquals( 1, ndarray.getFloat(0,0,0,1), 1e-10);
        assertEquals( 2, ndarray.getFloat(0,0,1,0), 1e-10);
        assertEquals( 3, ndarray.getFloat(0,0,1,1), 1e-10);
        assertEquals( 0, ndarray.getFloat(0,0,2,0), 1e-10);
        assertEquals( 5, ndarray.getFloat(0,0,2,1), 1e-10);
        assertEquals( 6, ndarray.getFloat(0,1,0,0), 1e-10);
        assertEquals( 7, ndarray.getFloat(0,1,0,1), 1e-10);
        assertEquals( 8, ndarray.getFloat(0,1,1,0), 1e-10);
        assertEquals( 9, ndarray.getFloat(0,1,1,1), 1e-10);
        assertEquals(10, ndarray.getFloat(0,1,2,0), 1e-10);
        assertEquals(11, ndarray.getFloat(0,1,2,1), 1e-10);
        assertEquals(12, ndarray.getFloat(0,2,0,0), 1e-10);
        assertEquals( 0, ndarray.getFloat(0,2,0,1), 1e-10);
        assertEquals(14, ndarray.getFloat(0,2,1,0), 1e-10);
        assertEquals(15, ndarray.getFloat(0,2,1,1), 1e-10);
        assertEquals(16, ndarray.getFloat(0,2,2,0), 1e-10);
        assertEquals(17, ndarray.getFloat(0,2,2,1), 1e-10);

        // 3,2,3
        ImageConverter second = new ImageConverter("test",3,2,3);
        ndarray = (TFloat32) second.convert(e,fmap).getMap().get("test");
        assertEquals( 0, ndarray.getFloat(0,0,0,0),1e-10);
        assertEquals( 1, ndarray.getFloat(0,0,0,1),1e-10);
        assertEquals( 2, ndarray.getFloat(0,0,0,2),1e-10);
        assertEquals( 3, ndarray.getFloat(0,0,1,0),1e-10);
        assertEquals( 0, ndarray.getFloat(0,0,1,1),1e-10);
        assertEquals( 5, ndarray.getFloat(0,0,1,2),1e-10);
        assertEquals( 6, ndarray.getFloat(0,1,0,0),1e-10);
        assertEquals( 7, ndarray.getFloat(0,1,0,1),1e-10);
        assertEquals( 8, ndarray.getFloat(0,1,0,2),1e-10);
        assertEquals( 9, ndarray.getFloat(0,1,1,0),1e-10);
        assertEquals(10, ndarray.getFloat(0,1,1,1),1e-10);
        assertEquals(11, ndarray.getFloat(0,1,1,2),1e-10);
        assertEquals(12, ndarray.getFloat(0,2,0,0),1e-10);
        assertEquals( 0, ndarray.getFloat(0,2,0,1),1e-10);
        assertEquals(14, ndarray.getFloat(0,2,0,2),1e-10);
        assertEquals(15, ndarray.getFloat(0,2,1,0),1e-10);
        assertEquals(16, ndarray.getFloat(0,2,1,1),1e-10);
        assertEquals(17, ndarray.getFloat(0,2,1,2),1e-10);

        // 2,3,3
        ImageConverter third = new ImageConverter("test",2,3,3);
        ndarray = (TFloat32) third.convert(e,fmap).getMap().get("test");
        assertEquals( 0, ndarray.getFloat(0,0,0,0),1e-10);
        assertEquals( 1, ndarray.getFloat(0,0,0,1),1e-10);
        assertEquals( 2, ndarray.getFloat(0,0,0,2),1e-10);
        assertEquals( 3, ndarray.getFloat(0,0,1,0),1e-10);
        assertEquals( 0, ndarray.getFloat(0,0,1,1),1e-10);
        assertEquals( 5, ndarray.getFloat(0,0,1,2),1e-10);
        assertEquals( 6, ndarray.getFloat(0,0,2,0),1e-10);
        assertEquals( 7, ndarray.getFloat(0,0,2,1),1e-10);
        assertEquals( 8, ndarray.getFloat(0,0,2,2),1e-10);
        assertEquals( 9, ndarray.getFloat(0,1,0,0),1e-10);
        assertEquals(10, ndarray.getFloat(0,1,0,1),1e-10);
        assertEquals(11, ndarray.getFloat(0,1,0,2),1e-10);
        assertEquals(12, ndarray.getFloat(0,1,1,0),1e-10);
        assertEquals( 0, ndarray.getFloat(0,1,1,1),1e-10);
        assertEquals(14, ndarray.getFloat(0,1,1,2),1e-10);
        assertEquals(15, ndarray.getFloat(0,1,2,0),1e-10);
        assertEquals(16, ndarray.getFloat(0,1,2,1),1e-10);
        assertEquals(17, ndarray.getFloat(0,1,2,2),1e-10);
    }

}

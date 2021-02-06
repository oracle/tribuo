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

package org.tribuo.math.la;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.impl.ArrayExample;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.util.function.DoubleUnaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for the DenseVector class.
 */
public class DenseVectorTest {

    private static DenseVector generateVectorA() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{1.0,2.0,0.0,0.0,3.0,4.0,0.0,0.0,5.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorB() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{-1.0,2.0,0.0,0.0,-3.0,4.0,0.0,0.0,-5.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorC() {
        //int[] indices = new int[]{1,5,6,7,9};
        double[] values = new double[]{0.0,2.0,0.0,0.0,0.0,3.0,4.0,5.0,0.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorASubB() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{2.0,0.0,0.0,0.0,6.0,0.0,0.0,0.0,10.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorASubC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,0.0,0.0,0.0,3.0,1.0,-4.0,-5.0,5.0,-6.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorBSubA() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{-2.0,0.0,0.0,0.0,-6.0,0.0,0.0,0.0,-10.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorBSubC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,0.0,0.0,0.0,-3.0,1.0,-4.0,-5.0,-5.0,-6.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorCSubA() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,0.0,0.0,0.0,-3.0,-1.0,4.0,5.0,-5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorCSubB() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,0.0,0.0,0.0,3.0,-1.0,4.0,5.0,5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateEmptyVector() {
        return new DenseVector(10);
    }

    private static DenseVector generateVectorAAddB() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{0.0,4.0,0.0,0.0,0.0,8.0,0.0,0.0,0.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorAAddC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,4.0,0.0,0.0,3.0,7.0,4.0,5.0,5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private static DenseVector generateVectorBAddC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,4.0,0.0,0.0,-3.0,7.0,4.0,5.0,-5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    @Test
    public void testReduction() {
        DenseVector a = generateVectorA();
        DenseVector b = generateVectorB();
        DenseVector c = generateVectorC();

        assertEquals(a.maxValue(),a.reduce(Double.MIN_VALUE, DoubleUnaryOperator.identity(), Double::max));
        assertEquals(b.maxValue(),b.reduce(Double.MIN_VALUE, DoubleUnaryOperator.identity(), Double::max));
        assertEquals(c.maxValue(),c.reduce(Double.MIN_VALUE, DoubleUnaryOperator.identity(), Double::max));

        assertEquals(a.minValue(),a.reduce(Double.MAX_VALUE, DoubleUnaryOperator.identity(), Double::min));
        assertEquals(b.minValue(),b.reduce(Double.MAX_VALUE, DoubleUnaryOperator.identity(), Double::min));
        assertEquals(c.minValue(),c.reduce(Double.MAX_VALUE, DoubleUnaryOperator.identity(), Double::min));

        assertEquals(a.sum(),a.reduce(0, DoubleUnaryOperator.identity(), Double::sum));
        assertEquals(b.sum(),b.reduce(0, DoubleUnaryOperator.identity(), Double::sum));
        assertEquals(c.sum(),c.reduce(0, DoubleUnaryOperator.identity(), Double::sum));

        assertEquals(a.sum(i -> i * i),a.reduce(0, i -> i * i, Double::sum));
        assertEquals(b.sum(Math::abs),b.reduce(0, Math::abs, Double::sum));
        assertEquals(c.sum(Math::exp),c.reduce(0, Math::exp, Double::sum));
    }

    @Test
    public void size() {
        DenseVector s = generateVectorA();
        assertEquals(10, s.size());
    }

    @Test
    public void overlappingDot() {
        DenseVector a = generateVectorA();
        DenseVector b = generateVectorB();

        assertEquals(a.dot(b), b.dot(a), 1e-10);
        assertEquals(-15.0, a.dot(b), 1e-10);
    }

    @Test
    public void dot() {
        DenseVector a = generateVectorA();
        DenseVector c = generateVectorC();

        assertEquals(a.dot(c),c.dot(a),1e-10);
        assertEquals(16.0, a.dot(c),1e-10);
    }

    @Test
    public void emptyDot() {
        DenseVector a = generateVectorA();
        DenseVector b = generateVectorB();
        DenseVector c = generateVectorC();
        DenseVector empty = generateEmptyVector();

        assertEquals(a.dot(empty),empty.dot(a),1e-10);
        assertEquals(0.0, a.dot(empty),1e-10);
        assertEquals(b.dot(empty),empty.dot(b),1e-10);
        assertEquals(0.0, b.dot(empty),1e-10);
        assertEquals(c.dot(empty),empty.dot(c),1e-10);
        assertEquals(0.0, c.dot(empty),1e-10);
    }

    @Test
    public void maxIndex() {
        DenseVector s = generateVectorB();
        assertEquals(5,s.indexOfMax());
    }

    @Test
    public void add() {
        DenseVector a = generateVectorA();
        DenseVector b = generateVectorB();
        DenseVector c = generateVectorC();
        DenseVector empty = generateEmptyVector();

        assertEquals(a,a.add(empty), "A + empty");
        assertEquals(b,b.add(empty), "B + empty");
        assertEquals(c,c.add(empty), "C + empty");

        assertEquals(scale(a,2.0),a.add(a), "A * 2");
        assertEquals(scale(b,2.0),b.add(b), "B * 2");
        assertEquals(scale(c,2.0),c.add(c), "C * 2");

        DenseVector aAddB = generateVectorAAddB();
        DenseVector aAddC = generateVectorAAddC();
        DenseVector bAddC = generateVectorBAddC();

        assertEquals(aAddB, a.add(b), "A + B");
        assertEquals(aAddC, a.add(c), "A + C");
        assertEquals(aAddB, b.add(a), "B + A");
        assertEquals(bAddC, b.add(c), "B + C");
        assertEquals(aAddC, c.add(a), "C + A");
        assertEquals(bAddC, c.add(b), "C + B");
    }

    @Test
    public void subtract() {
        DenseVector a = generateVectorA();
        DenseVector b = generateVectorB();
        DenseVector c = generateVectorC();
        DenseVector empty = generateEmptyVector();

        assertEquals(a,a.subtract(empty), "A - empty");
        assertEquals(invert(a),empty.subtract(a), "empty - A");
        assertEquals(b,b.subtract(empty), "B - empty");
        assertEquals(invert(b),empty.subtract(b), "empty - B");
        assertEquals(c,c.subtract(empty), "C - empty");
        assertEquals(invert(c),empty.subtract(c), "empty - C");

        assertEquals(zero(a),a.subtract(a), "A - A");
        assertEquals(zero(b),b.subtract(b), "B - B");
        assertEquals(zero(c),c.subtract(c), "C - C");

        DenseVector aSubB = generateVectorASubB();
        DenseVector aSubC = generateVectorASubC();
        DenseVector bSubA = generateVectorBSubA();
        DenseVector bSubC = generateVectorBSubC();
        DenseVector cSubA = generateVectorCSubA();
        DenseVector cSubB = generateVectorCSubB();

        assertEquals(aSubB, a.subtract(b), "A - B");
        assertEquals(aSubC, a.subtract(c), "A - C");
        assertEquals(bSubA, b.subtract(a), "B - A");
        assertEquals(bSubC, b.subtract(c), "B - C");
        assertEquals(cSubA, c.subtract(a), "C - A");
        assertEquals(cSubB, c.subtract(b), "C - B");
    }

    @Test
    public void testExampleCreation() {
        ImmutableFeatureMap fmap = Helpers.mkFeatureMap("F0","F1","F2","F3","F4");

        MockOutput output = MockOutputFactory.UNKNOWN_TEST_OUTPUT;

        // Check without bias
        Example<MockOutput> dense = Helpers.mkExample(output,"F0","F1","F2","F3","F4");
        DenseVector vector = DenseVector.createDenseVector(dense,fmap,false);
        Assertions.assertEquals(5,vector.size());
        Assertions.assertEquals(5,vector.numActiveElements());

        // Check bias
        vector = DenseVector.createDenseVector(dense,fmap,true);
        Assertions.assertEquals(6,vector.size());
        Assertions.assertEquals(6,vector.numActiveElements());

        // Check sparse is made dense
        Example<MockOutput> sparse = Helpers.mkExample(output,"F1","F3");
        vector = DenseVector.createDenseVector(sparse,fmap,false);
        Assertions.assertEquals(2,sparse.size());
        Assertions.assertEquals(5,vector.size());
        Assertions.assertEquals(5,vector.numActiveElements());

        // Check extra features are ignored
        Example<MockOutput> extraFeatures = Helpers.mkExample(output,"F0","F1","F2","F3","F4","F5");
        vector = DenseVector.createDenseVector(extraFeatures,fmap,false);
        Assertions.assertEquals(5,vector.size());
        Assertions.assertEquals(5,vector.numActiveElements());

        // Check the right values are present
        Example<MockOutput> values = new ArrayExample<>(output,
                new String[]{"F0","F1","F2","F3","F4"},
                new double[]{-1,2,-3,4,-5});
        vector = DenseVector.createDenseVector(values,fmap,true);
        Assertions.assertEquals(6,vector.size());
        Assertions.assertEquals(6,vector.numActiveElements());
        Assertions.assertEquals(-1,vector.get(0));
        Assertions.assertEquals(2,vector.get(1));
        Assertions.assertEquals(-3,vector.get(2));
        Assertions.assertEquals(4,vector.get(3));
        Assertions.assertEquals(-5,vector.get(4));
        Assertions.assertEquals(1,vector.get(5));

        // Check empty example throws
        Example<MockOutput> empty = Helpers.mkExample(output,new String[0]);
        Assertions.assertThrows(IllegalArgumentException.class,() -> DenseVector.createDenseVector(empty,fmap,true));
        Assertions.assertThrows(IllegalArgumentException.class,() -> DenseVector.createDenseVector(empty,fmap,false));

        // Check example with no feature overlap throws
        Example<MockOutput> noOverlap = Helpers.mkExample(output, "A0","A1");
        Assertions.assertThrows(IllegalArgumentException.class,() -> DenseVector.createDenseVector(noOverlap,fmap,true));
        Assertions.assertThrows(IllegalArgumentException.class,() -> DenseVector.createDenseVector(noOverlap,fmap,false));

        // Check NaN valued feature throws
        Example<MockOutput> nanFeatures = new ArrayExample<MockOutput>(output,new String[]{"F0"},new double[]{Double.NaN});
        Assertions.assertThrows(IllegalArgumentException.class,() -> DenseVector.createDenseVector(nanFeatures,fmap,true));
        Assertions.assertThrows(IllegalArgumentException.class,() -> DenseVector.createDenseVector(nanFeatures,fmap,false));
    }

    public static DenseVector invert(DenseVector input) {
        return scale(input,-1.0);
    }

    public static DenseVector zero(DenseVector input) {
        return scale(input,0.0);
    }

    public static DenseVector scale(DenseVector input, double scale) {
        DenseVector inverse = new DenseVector(input);
        inverse.scaleInPlace(scale);
        return inverse;
    }
}

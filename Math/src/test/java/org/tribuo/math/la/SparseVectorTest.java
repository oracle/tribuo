/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import org.tribuo.CategoricalIDInfo;
import org.tribuo.CategoricalInfo;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.MutableFeatureMap;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.ListExample;
import org.tribuo.math.protos.SparseTensorProto;
import org.tribuo.math.protos.TensorProto;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.util.Util;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Tests for the SparseVector class.
 */
public class SparseVectorTest {

    private SparseVector generateVectorA() {
        int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{1.0,2.0,3.0,4.0,5.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorB() {
        int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{-1.0,2.0,-3.0,4.0,-5.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorC() {
        int[] indices = new int[]{1,5,6,7,9};
        double[] values = new double[]{2.0,3.0,4.0,5.0,6.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorASubB() {
        int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{2.0,0.0,6.0,0.0,10.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorASubC() {
        int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,0.0,3.0,1.0,-4.0,-5.0,5.0,-6.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorBSubA() {
        int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{-2.0,0.0,-6.0,0.0,-10.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorBSubC() {
        int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,0.0,-3.0,1.0,-4.0,-5.0,-5.0,-6.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorCSubA() {
        int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,0.0,-3.0,-1.0,4.0,5.0,-5.0,6.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorCSubB() {
        int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,0.0,3.0,-1.0,4.0,5.0,5.0,6.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateEmptyVector() {
        int[] indices = new int[0];
        double[] values = new double[0];
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorAAddB() {
        int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{0.0,4.0,0.0,8.0,0.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorAAddC() {
        int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,4.0,3.0,7.0,4.0,5.0,5.0,6.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    private SparseVector generateVectorBAddC() {
        int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,4.0,-3.0,7.0,4.0,5.0,-5.0,6.0};
        return SparseVector.createSparseVector(10,indices,values);
    }

    @Test
    public void testReduction() {
        SparseVector a = generateVectorA();
        SparseVector b = generateVectorB();
        SparseVector c = generateVectorC();

        assertEquals(a.maxValue(),a.reduce(Double.MIN_VALUE, DoubleUnaryOperator.identity(), Double::max));
        assertEquals(b.maxValue(),b.reduce(Double.MIN_VALUE, DoubleUnaryOperator.identity(), Double::max));
        assertEquals(c.maxValue(),c.reduce(Double.MIN_VALUE, DoubleUnaryOperator.identity(), Double::max));

        assertEquals(0.0,a.reduce(Double.MAX_VALUE, DoubleUnaryOperator.identity(), Double::min));
        assertEquals(-5.0,b.reduce(Double.MAX_VALUE, DoubleUnaryOperator.identity(), Double::min));
        assertEquals(0.0,c.reduce(Double.MAX_VALUE, DoubleUnaryOperator.identity(), Double::min));

        assertEquals(a.sum(),a.reduce(0, DoubleUnaryOperator.identity(), Double::sum));
        assertEquals(b.sum(),b.reduce(0, DoubleUnaryOperator.identity(), Double::sum));
        assertEquals(c.sum(),c.reduce(0, DoubleUnaryOperator.identity(), Double::sum));
    }

    @Test
    public void size() {
        SparseVector s = generateVectorA();
        assertEquals(10, s.size());
    }

    @Test
    public void activeElements() {
        SparseVector s = generateVectorA();
        assertEquals(5, s.numActiveElements());
    }

    @Test
    public void overlappingDot() {
        SparseVector a = generateVectorA();
        SparseVector b = generateVectorB();

        assertEquals(a.dot(b), b.dot(a), 1e-10);
        assertEquals(-15.0, a.dot(b), 1e-10);
    }

    @Test
    public void dot() {
        SparseVector a = generateVectorA();
        SparseVector c = generateVectorC();

        assertEquals(a.dot(c),c.dot(a),1e-10);
        assertEquals(16.0, a.dot(c),1e-10);
    }

    @Test
    public void emptyDot() {
        SparseVector a = generateVectorA();
        SparseVector b = generateVectorB();
        SparseVector c = generateVectorC();
        SparseVector empty = generateEmptyVector();

        assertEquals(a.dot(empty),empty.dot(a),1e-10);
        assertEquals(0.0, a.dot(empty),1e-10);
        assertEquals(b.dot(empty),empty.dot(b),1e-10);
        assertEquals(0.0, b.dot(empty),1e-10);
        assertEquals(c.dot(empty),empty.dot(c),1e-10);
        assertEquals(0.0, c.dot(empty),1e-10);
    }

    @Test
    public void maxIndex() {
        SparseVector s = generateVectorB();
        assertEquals(5,s.indexOfMax());
    }

    @Test
    public void difference() {
        SparseVector a = generateVectorA();
        SparseVector b = generateVectorB();
        SparseVector c = generateVectorC();

        assertArrayEquals(a.difference(b), new int[0]);
        assertArrayEquals(a.difference(c), new int[]{0,4,8});
        assertArrayEquals(c.difference(a), new int[]{6,7,9});
    }

    @Test
    public void intersection() {
        SparseVector a = generateVectorA();
        SparseVector b = generateVectorB();
        SparseVector c = generateVectorC();
        SparseVector empty = generateEmptyVector();

        assertArrayEquals(a.intersection(b), new int[]{0,1,4,5,8});
        assertArrayEquals(a.intersection(c), new int[]{1,5});
        assertArrayEquals(a.intersection(empty), new int[0]);
    }

    @Test
    public void add() {
        SparseVector a = generateVectorA();
        SparseVector b = generateVectorB();
        SparseVector c = generateVectorC();
        SparseVector empty = generateEmptyVector();

        assertEquals(a,a.add(empty), "A + empty");
        assertEquals(b,b.add(empty), "B + empty");
        assertEquals(c,c.add(empty), "C + empty");

        assertEquals(scale(a,2.0),a.add(a), "A * 2");
        assertEquals(scale(b,2.0),b.add(b), "B * 2");
        assertEquals(scale(c,2.0),c.add(c), "C * 2");

        SparseVector aAddB = generateVectorAAddB();
        SparseVector aAddC = generateVectorAAddC();
        SparseVector bAddC = generateVectorBAddC();

        assertEquals(aAddB, a.add(b), "A + B");
        assertEquals(aAddC, a.add(c), "A + C");
        assertEquals(aAddB, b.add(a), "B + A");
        assertEquals(bAddC, b.add(c), "B + C");
        assertEquals(aAddC, c.add(a), "C + A");
        assertEquals(bAddC, c.add(b), "C + B");
    }

    @Test
    public void subtract() {
        SparseVector a = generateVectorA();
        SparseVector b = generateVectorB();
        SparseVector c = generateVectorC();
        SparseVector empty = generateEmptyVector();

        assertEquals(a,a.subtract(empty), "A - empty");
        assertEquals(invert(a),empty.subtract(a), "empty - A");
        assertEquals(b,b.subtract(empty), "B - empty");
        assertEquals(invert(b),empty.subtract(b), "empty - B");
        assertEquals(c,c.subtract(empty), "C - empty");
        assertEquals(invert(c),empty.subtract(c), "empty - C");

        assertEquals(zero(a),a.subtract(a), "A - A");
        assertEquals(zero(b),b.subtract(b), "B - B");
        assertEquals(zero(c),c.subtract(c), "C - C");

        SparseVector aSubB = generateVectorASubB();
        SparseVector aSubC = generateVectorASubC();
        SparseVector bSubA = generateVectorBSubA();
        SparseVector bSubC = generateVectorBSubC();
        SparseVector cSubA = generateVectorCSubA();
        SparseVector cSubB = generateVectorCSubB();

        assertEquals(aSubB, a.subtract(b), "A - B");
        assertEquals(aSubC, a.subtract(c), "A - C");
        assertEquals(bSubA, b.subtract(a), "B - A");
        assertEquals(bSubC, b.subtract(c), "B - C");
        assertEquals(cSubA, c.subtract(a), "C - A");
        assertEquals(cSubB, c.subtract(b), "C - B");
    }

    public static SparseVector invert(SparseVector input) {
        return scale(input,-1.0);
    }

    public static SparseVector zero(SparseVector input) {
        return scale(input,0.0);
    }

    public static SparseVector scale(SparseVector input, double scale) {
        SparseVector inverse = input.copy();
        inverse.scaleInPlace(scale);
        return inverse;
    }

    @Test
    public void testTranspose() {
        SparseVector[] input = new SparseVector[3];
        input[0] = generateVectorA();
        input[1] = generateVectorB();
        input[2] = generateVectorC();

        SparseVector[] transpose = SparseVector.transpose(input);
        SparseVector[] inputPrime = SparseVector.transpose(transpose);

        assertEquals(input.length,inputPrime.length);

        for (int i = 0; i < input.length; i++) {
            assertEquals(input[i],inputPrime[i]);
        }
    }

    @Test
    public void testExampleTranspose() {
        ArrayList<Example<MockOutput>> examples = new ArrayList<>();
        SparseVector[] vectors = new SparseVector[4];

        MockOutput output = new MockOutput("test");

        Example<MockOutput> e = new ArrayExample<>(output,new String[]{"A","B","C","D","E"},new double[]{1,2,3,4,5});
        examples.add(e);
        e = new ArrayExample<>(output,new String[]{"A","C","D"},new double[]{10,30,40});
        examples.add(e);
        e = new ArrayExample<>(output,new String[]{"A","C","D"},new double[]{100,300,400});
        examples.add(e);
        e = new ArrayExample<>(output,new String[]{"A","C","D"},new double[]{0.1,0.3,0.4});
        examples.add(e);

        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(),new MockOutputFactory());
        dataset.addAll(examples);
        ImmutableFeatureMap fMap = dataset.getFeatureIDMap();
        vectors[0] = SparseVector.createSparseVector(examples.get(0),fMap,false);
        vectors[1] = SparseVector.createSparseVector(examples.get(1),fMap,false);
        vectors[2] = SparseVector.createSparseVector(examples.get(2),fMap,false);
        vectors[3] = SparseVector.createSparseVector(examples.get(3),fMap,false);

        SparseVector[] transposedExamples = SparseVector.transpose(dataset);
        SparseVector[] transposedVectors = SparseVector.transpose(vectors);

        assertEquals(5,transposedExamples.length);
        assertEquals(transposedExamples.length,transposedVectors.length);

        for (int i = 0; i < transposedExamples.length; i++) {
            assertEquals(transposedExamples[i],transposedVectors[i]);
        }
    }

    @Test
    public void duplicateFeatureIDs() throws Exception {
        ImmutableFeatureMap fmap = new TestMap();

        Example<MockOutput> collision = generateExample(new String[]{"FOO","BAR","BAZ","QUUX"},new double[]{1.0,2.2,3.3,4.4});
        SparseVector testCollisionVec = SparseVector.createSparseVector(3,new int[]{0,1,2},new double[]{4.3,2.2,4.4});
        SparseVector collisionVec = SparseVector.createSparseVector(collision,fmap,false);
        assertEquals(testCollisionVec,collisionVec);

        Example<MockOutput> fakecollision = generateExample(new String[]{"BAR","BAZ","QUUX"},new double[]{2.2,3.3,4.4});
        SparseVector testFakeCollisionVec = SparseVector.createSparseVector(3,new int[]{0,1,2},new double[]{3.3,2.2,4.4});
        SparseVector fakeCollisionVec = SparseVector.createSparseVector(fakecollision,fmap,false);
        assertEquals(testFakeCollisionVec,fakeCollisionVec);
    }

    @Test
    public void differenceRandomised() {
        Random rng = new Random(1);

        int iterations = 1000;
        int vectorLength = 200;
        int dimension = 1000;

        int[] perm;

        double[] values = new double[vectorLength];
        Arrays.fill(values,2.0);

        for (int i = 0; i < iterations; i++) {
            perm = Util.randperm(dimension, rng);
            int[] firstIndices = new int[vectorLength];
            System.arraycopy(perm, 0, firstIndices, 0, vectorLength);
            Arrays.sort(firstIndices);

            perm = Util.randperm(dimension, rng);
            int[] secondIndices = new int[vectorLength];
            System.arraycopy(perm, 0, secondIndices, 0, vectorLength);
            Arrays.sort(secondIndices);

            SparseVector first = SparseVector.createSparseVector(dimension, firstIndices, values);
            SparseVector second = SparseVector.createSparseVector(dimension, secondIndices, values);

            int[] diff = first.difference(second);
            int[] otherDiff = second.difference(first);

            int[] slowDiff = slowDifference(firstIndices, secondIndices);
            int[] slowOtherDiff = slowDifference(secondIndices, firstIndices);
            assertArrayEquals(slowDiff, diff);
            assertArrayEquals(slowOtherDiff, otherDiff);
        }
    }

    @Test
    public void intersectionRandomised() {
        Random rng = new Random(1);

        int iterations = 1000;
        int vectorLength = 200;
        int dimension = 1000;

        int[] perm;

        double[] values = new double[vectorLength];
        Arrays.fill(values,2.0);

        for (int i = 0; i < iterations; i++) {
            perm = Util.randperm(dimension, rng);
            int[] firstIndices = new int[vectorLength];
            System.arraycopy(perm, 0, firstIndices, 0, vectorLength);
            Arrays.sort(firstIndices);

            perm = Util.randperm(dimension, rng);
            int[] secondIndices = new int[vectorLength];
            System.arraycopy(perm, 0, secondIndices, 0, vectorLength);
            Arrays.sort(secondIndices);

            SparseVector first = SparseVector.createSparseVector(dimension, firstIndices, values);
            SparseVector second = SparseVector.createSparseVector(dimension, secondIndices, values);

            int[] intersect = first.intersection(second);
            int[] otherDiff = second.intersection(first);

            int[] slowDiff = slowIntersection(firstIndices, secondIndices);
            int[] slowOtherDiff = slowIntersection(secondIndices, firstIndices);
            assertArrayEquals(slowDiff, intersect);
            assertArrayEquals(slowOtherDiff, otherDiff);
        }
    }

    @Test
    public void distanceRandomised() {
        Random rng = new Random(1);

        int iterations = 1000;
        int vectorLength = 200;
        int dimension = 1000;

        int[] perm;

        double[] values = new double[vectorLength];
        for (int i = 0; i < values.length; i++) {
            values[i] = i;
        }

        for (int i = 0; i < iterations; i++) {
            perm = Util.randperm(dimension, rng);
            int[] firstIndices = new int[vectorLength];
            System.arraycopy(perm, 0, firstIndices, 0, vectorLength);
            Arrays.sort(firstIndices);

            perm = Util.randperm(dimension, rng);
            int[] secondIndices = new int[vectorLength];
            System.arraycopy(perm, 0, secondIndices, 0, vectorLength);
            Arrays.sort(secondIndices);

            SparseVector first = SparseVector.createSparseVector(dimension, firstIndices, values);
            SparseVector second = SparseVector.createSparseVector(dimension, secondIndices, values);

            double score = first.l1Distance(second);
            double otherScore = second.l1Distance(first);

            double slowDistance = slowl1Distance(dimension, firstIndices, values, secondIndices, values);
            assertEquals(slowDistance, score, 1e-10);

            assertEquals(score, otherScore, 1e-10);
        }
    }

    @Test
    public void dotRandomised() {
        Random rng = new Random(1);

        int iterations = 1000;
        int vectorLength = 200;
        int dimension = 1000;

        int[] perm;

        double[] values = new double[vectorLength];
        Arrays.fill(values,2.0);

        for (int i = 0; i < iterations; i++) {
            perm = Util.randperm(dimension, rng);
            int[] firstIndices = new int[vectorLength];
            System.arraycopy(perm, 0, firstIndices, 0, vectorLength);
            Arrays.sort(firstIndices);

            perm = Util.randperm(dimension, rng);
            int[] secondIndices = new int[vectorLength];
            System.arraycopy(perm, 0, secondIndices, 0, vectorLength);
            Arrays.sort(secondIndices);

            SparseVector first = SparseVector.createSparseVector(dimension, firstIndices, values);
            SparseVector second = SparseVector.createSparseVector(dimension, secondIndices, values);

            double score = first.dot(second);
            double otherScore = second.dot(first);

            double slowDot = slowDot(firstIndices, values, secondIndices, values);
            assertEquals(slowDot, score, 1e-10);

            assertEquals(score, otherScore, 1e-10);
        }
    }

    @Test
    public void hadamardRandomized() {
        Random rng = new Random(1);

        int iterations = 1000;
        int vectorLength = 200;
        int dimension = 1000;

        int[] perm;

        double[] values = new double[vectorLength];
        for (int i = 0; i < values.length; i++) {
            values[i] = i*3;
        }

        for (int i = 0; i < iterations; i++) {
            perm = Util.randperm(dimension, rng);
            int[] firstIndices = new int[vectorLength];
            System.arraycopy(perm, 0, firstIndices, 0, vectorLength);
            Arrays.sort(firstIndices);

            perm = Util.randperm(dimension, rng);
            int[] secondIndices = new int[vectorLength];
            System.arraycopy(perm, 0, secondIndices, 0, vectorLength);
            Arrays.sort(secondIndices);

            SparseVector firstA = SparseVector.createSparseVector(dimension, firstIndices, Arrays.copyOf(values,values.length));
            SparseVector firstB = SparseVector.createSparseVector(dimension, firstIndices, Arrays.copyOf(values,values.length));
            SparseVector secondA = SparseVector.createSparseVector(dimension, secondIndices, Arrays.copyOf(values,values.length));
            SparseVector secondB = SparseVector.createSparseVector(dimension, secondIndices, Arrays.copyOf(values,values.length));

            firstA.hadamardProductInPlace(secondA);
            secondB.hadamardProductInPlace(firstB);

            double[] slowHadamardA = slowHadamard(firstIndices, values, secondIndices, values);
            double[] slowHadamardB = slowHadamard(secondIndices, values, firstIndices, values);
            assertArrayEquals(slowHadamardA, firstA.values, 1e-10);
            assertArrayEquals(slowHadamardB, secondB.values, 1e-10);
        }
    }

    @Test
    public void addInPlaceRandomized() {
        Random rng = new Random(1);

        int iterations = 1000;
        int vectorLength = 200;
        int dimension = 1000;

        int[] perm;

        double[] values = new double[vectorLength];
        for (int i = 0; i < values.length; i++) {
            values[i] = i*3;
        }

        for (int i = 0; i < iterations; i++) {
            perm = Util.randperm(dimension, rng);
            int[] firstIndices = new int[vectorLength];
            System.arraycopy(perm, 0, firstIndices, 0, vectorLength);
            Arrays.sort(firstIndices);

            perm = Util.randperm(dimension, rng);
            int[] secondIndices = new int[vectorLength];
            System.arraycopy(perm, 0, secondIndices, 0, vectorLength);
            Arrays.sort(secondIndices);

            SparseVector firstA = SparseVector.createSparseVector(dimension, firstIndices, Arrays.copyOf(values,values.length));
            SparseVector firstB = SparseVector.createSparseVector(dimension, firstIndices, Arrays.copyOf(values,values.length));
            SparseVector secondA = SparseVector.createSparseVector(dimension, secondIndices, Arrays.copyOf(values,values.length));
            SparseVector secondB = SparseVector.createSparseVector(dimension, secondIndices, Arrays.copyOf(values,values.length));

            firstA.intersectAndAddInPlace(secondA);
            secondB.intersectAndAddInPlace(firstB);

            double[] slowAddA = slowAdd(firstIndices, values, secondIndices, values);
            double[] slowAddB = slowAdd(secondIndices, values, firstIndices, values);
            assertArrayEquals(slowAddA, firstA.values, 1e-10);
            assertArrayEquals(slowAddB, secondB.values, 1e-10);
        }
    }

    @Test
    public void createFromExample() {
        MutableFeatureMap featureMap = new MutableFeatureMap();
        featureMap.add("A",1);
        featureMap.add("B",1);
        featureMap.add("C",1);
        featureMap.add("D",1);
        featureMap.add("E",1);
        featureMap.add("F",1);
        featureMap.add("G",1);
        ImmutableFeatureMap immutableFeatureMap = new ImmutableFeatureMap(featureMap);
        assertEquals(7,immutableFeatureMap.size());

        MockOutput mock = new MockOutput("MOCK");
        ArrayExample<MockOutput> example = new ArrayExample<>(mock,new String[]{"A","B","C"},new double[]{1,2,3});
        SparseVector vector = SparseVector.createSparseVector(example,immutableFeatureMap,false);
        assertEquals(3,vector.numActiveElements());
        assertEquals(immutableFeatureMap.size(),vector.size());
        assertEquals(1.0,vector.get(0));
        assertEquals(2.0,vector.get(1));
        assertEquals(3.0,vector.get(2));
        assertEquals(0.0,vector.get(3));
        assertEquals(0.0,vector.get(4));
        assertEquals(0.0,vector.get(5));
        assertEquals(0.0,vector.get(6));
        assertEquals(0.0,vector.get(7));

        example = new ArrayExample<>(mock,new String[]{"A","AA","BB","CC"},new double[]{1,2,3,4});
        vector = SparseVector.createSparseVector(example,immutableFeatureMap,false);
        assertEquals(1,vector.numActiveElements());
        assertEquals(immutableFeatureMap.size(),vector.size());
        assertEquals(1.0,vector.get(0));
        assertEquals(0.0,vector.get(1));
        assertEquals(0.0,vector.get(2));
        assertEquals(0.0,vector.get(3));
        assertEquals(0.0,vector.get(4));
        assertEquals(0.0,vector.get(5));
        assertEquals(0.0,vector.get(6));
        assertEquals(0.0,vector.get(7));

        example = new ArrayExample<>(mock,new String[]{"A","AA","BB","CC"},new double[]{1,2,3,4});
        vector = SparseVector.createSparseVector(example,immutableFeatureMap,true);
        assertEquals(2,vector.numActiveElements());
        assertEquals(immutableFeatureMap.size()+1,vector.size()); // due to bias
        assertEquals(1.0,vector.get(0));
        assertEquals(0.0,vector.get(1));
        assertEquals(0.0,vector.get(2));
        assertEquals(0.0,vector.get(3));
        assertEquals(0.0,vector.get(4));
        assertEquals(0.0,vector.get(5));
        assertEquals(0.0,vector.get(6));
        assertEquals(1.0,vector.get(7));
    }

    private static double[] slowHadamard(int[] firstIndices, double[] firstValues, int[] secondIndices, double[] secondValues) {
        double[] output = new double[firstValues.length];
        for (int i = 0; i < firstIndices.length; i++) {
            int curIndex = firstIndices[i];
            int secondPos = Arrays.binarySearch(secondIndices,curIndex);
            if (secondPos > -1) {
                output[i] = firstValues[i] * secondValues[secondPos];
            } else {
                output[i] = firstValues[i];
            }
        }
        return output;
    }

    private static double[] slowAdd(int[] firstIndices, double[] firstValues, int[] secondIndices, double[] secondValues) {
        double[] output = new double[firstValues.length];
        for (int i = 0; i < firstIndices.length; i++) {
            int curIndex = firstIndices[i];
            int secondPos = Arrays.binarySearch(secondIndices,curIndex);
            if (secondPos > -1) {
                output[i] = firstValues[i] + secondValues[secondPos];
            } else {
                output[i] = firstValues[i];
            }
        }
        return output;
    }

    private static double slowDot(int[] firstIndices, double[] firstValues, int[] secondIndices, double[] secondValues) {
        double sum = 0.0;
        for (int i = 0; i < firstIndices.length; i++) {
            int curIndex = firstIndices[i];
            int secondPos = Arrays.binarySearch(secondIndices,curIndex);
            if (secondPos > -1) {
                assertEquals(curIndex,secondIndices[secondPos]);
                sum += firstValues[i] * secondValues[secondPos];
            }
        }

        return sum;
    }

    private static double slowl1Distance(int size, int[] firstIndices, double[] firstValues, int[] secondIndices, double[] secondValues) {
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            int firstIndex = Arrays.binarySearch(firstIndices,i);
            int secondIndex = Arrays.binarySearch(secondIndices,i);
            if ((firstIndex > -1) && (secondIndex > -1)) {
                sum += Math.abs(firstValues[firstIndex] - secondValues[secondIndex]);
            } else if (firstIndex > -1) {
                sum += Math.abs(firstValues[firstIndex]);
            } else if (secondIndex > -1) {
                sum += Math.abs(secondValues[secondIndex]);
            }
        }

        return sum;
    }

    private static int[] slowDifference(int[] firstIndices, int[] secondIndices) {
        ArrayList<Integer> diffIndices = new ArrayList<>();

        for (int i = 0; i < firstIndices.length; i++) {
            int curIndex = firstIndices[i];
            int secondPos = Arrays.binarySearch(secondIndices,curIndex);
            if (secondPos < 0) {
                diffIndices.add(curIndex);
            }
        }

        return Util.toPrimitiveInt(diffIndices);
    }

    private static int[] slowIntersection(int[] firstIndices, int[] secondIndices) {
        ArrayList<Integer> intersectIndices = new ArrayList<>();

        for (int i = 0; i < firstIndices.length; i++) {
            int curIndex = firstIndices[i];
            int secondPos = Arrays.binarySearch(secondIndices,curIndex);
            if (secondPos > -1) {
                intersectIndices.add(curIndex);
            }
        }

        return Util.toPrimitiveInt(intersectIndices);
    }

    @Test
    public void serializationTest() {
        SparseVector a = generateVectorA();
        TensorProto proto = a.serialize();
        Tensor deser = Tensor.deserialize(proto);
        assertEquals(a, deser);

        SparseVector empty = new SparseVector(10, new int[0], new double[0]);
        proto = empty.serialize();
        deser = Tensor.deserialize(proto);
        assertEquals(empty, deser);

        SparseVector full = new SparseVector(10, new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 5);
        proto = full.serialize();
        deser = Tensor.deserialize(proto);
        assertEquals(full, deser);

    }

    @Test
    public void serializationValidationTest() {
        String className = SparseVector.class.getName();
        TensorProto negSize = makeMalformedSparseProto(className,new int[]{-1}, 2, new int[]{0,1}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(negSize);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto negNonZero = makeMalformedSparseProto(className,new int[]{5}, -1, new int[]{0,3}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(negNonZero);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto nonZeroMismatch = makeMalformedSparseProto(className,new int[]{5}, 3, new int[]{0,3}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(nonZeroMismatch);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto invalidIndices = makeMalformedSparseProto(className, new int[]{5}, 2, new int[]{0,-1}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(invalidIndices);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto valueIndicesMismatch = makeMalformedSparseProto(className, new int[]{5}, 2, new int[]{0,3}, new double[1]);
        try {
            Tensor deser = Tensor.deserialize(valueIndicesMismatch);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto matrix = makeMalformedSparseProto(className, new int[]{5,5}, 2, new int[]{0,0,3,2}, new double[]{1,2});
        try {
            Tensor deser = Tensor.deserialize(matrix);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }
    }

    static TensorProto makeMalformedSparseProto(String className, int[] size, int numNonZero, int[] indices, double[] values) {
        TensorProto.Builder builder = TensorProto.newBuilder();

        builder.setVersion(0);
        builder.setClassName(className);

        SparseTensorProto.Builder dataBuilder = SparseTensorProto.newBuilder();
        dataBuilder.addAllDimensions(Arrays.stream(size).boxed().collect(Collectors.toList()));
        ByteBuffer indicesBuffer = ByteBuffer.allocate(indices.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer intBuffer = indicesBuffer.asIntBuffer();
        intBuffer.put(indices);
        intBuffer.rewind();
        ByteBuffer valuesBuffer = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer doubleBuffer = valuesBuffer.asDoubleBuffer();
        doubleBuffer.put(values);
        doubleBuffer.rewind();
        dataBuilder.setIndices(ByteString.copyFrom(indicesBuffer));
        dataBuilder.setValues(ByteString.copyFrom(valuesBuffer));
        dataBuilder.setNumNonZero(numNonZero);
        builder.setSerializedData(Any.pack(dataBuilder.build()));

        return builder.build();
    }

    private static Example<MockOutput> generateExample(String[] names, double[] values) {
        Example<MockOutput> e = new ListExample<>(new MockOutput("MONKEYS"));
        for (int i = 0; i < names.length; i++) {
            e.add(new Feature(names[i],values[i]));
        }
        return e;
    }

    private static class TestMap extends ImmutableFeatureMap {
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

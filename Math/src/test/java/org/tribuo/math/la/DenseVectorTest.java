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

package org.tribuo.math.la;


import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for the DenseVector class.
 */
public class DenseVectorTest {

    private DenseVector generateVectorA() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{1.0,2.0,0.0,0.0,3.0,4.0,0.0,0.0,5.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorB() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{-1.0,2.0,0.0,0.0,-3.0,4.0,0.0,0.0,-5.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorC() {
        //int[] indices = new int[]{1,5,6,7,9};
        double[] values = new double[]{0.0,2.0,0.0,0.0,0.0,3.0,4.0,5.0,0.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorASubB() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{2.0,0.0,0.0,0.0,6.0,0.0,0.0,0.0,10.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorASubC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,0.0,0.0,0.0,3.0,1.0,-4.0,-5.0,5.0,-6.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorBSubA() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{-2.0,0.0,0.0,0.0,-6.0,0.0,0.0,0.0,-10.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorBSubC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,0.0,0.0,0.0,-3.0,1.0,-4.0,-5.0,-5.0,-6.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorCSubA() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,0.0,0.0,0.0,-3.0,-1.0,4.0,5.0,-5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorCSubB() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,0.0,0.0,0.0,3.0,-1.0,4.0,5.0,5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateEmptyVector() {
        return new DenseVector(10);
    }

    private DenseVector generateVectorAAddB() {
        //int[] indices = new int[]{0,1,4,5,8};
        double[] values = new double[]{0.0,4.0,0.0,0.0,0.0,8.0,0.0,0.0,0.0,0.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorAAddC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{1.0,4.0,0.0,0.0,3.0,7.0,4.0,5.0,5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    private DenseVector generateVectorBAddC() {
        //int[] indices = new int[]{0,1,4,5,6,7,8,9};
        double[] values = new double[]{-1.0,4.0,0.0,0.0,-3.0,7.0,4.0,5.0,-5.0,6.0};
        return DenseVector.createDenseVector(values);
    }

    @Test
    public void size() throws Exception {
        DenseVector s = generateVectorA();
        assertEquals(10, s.size());
    }

    @Test
    public void overlappingDot() throws Exception {
        DenseVector a = generateVectorA();
        DenseVector b = generateVectorB();

        assertEquals(a.dot(b), b.dot(a), 1e-10);
        assertEquals(-15.0, a.dot(b), 1e-10);
    }

    @Test
    public void dot() throws Exception {
        DenseVector a = generateVectorA();
        DenseVector c = generateVectorC();

        assertEquals(a.dot(c),c.dot(a),1e-10);
        assertEquals(16.0, a.dot(c),1e-10);
    }

    @Test
    public void emptyDot() throws Exception {
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
    public void maxIndex() throws Exception {
        DenseVector s = generateVectorB();
        assertEquals(5,s.indexOfMax());
    }

    @Test
    public void add() throws Exception {
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
    public void subtract() throws Exception {
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

    public DenseVector invert(DenseVector input) {
        return scale(input,-1.0);
    }

    public DenseVector zero(DenseVector input) {
        return scale(input,0.0);
    }

    public DenseVector scale(DenseVector input, double scale) {
        DenseVector inverse = new DenseVector(input);
        inverse.scaleInPlace(scale);
        return inverse;
    }
}

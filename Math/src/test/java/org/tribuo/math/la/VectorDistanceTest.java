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
 *
 */
public class VectorDistanceTest {

    public DenseVector generateDenseA() {
        return DenseVector.createDenseVector(new double[]{0,1,2,3,4,5,6,7,8,9});
    }

    public DenseVector generateDenseB() {
        return DenseVector.createDenseVector(new double[]{9,8,7,6,5,4,3,2,1,0});
    }

    public SparseVector generateSparseC() {
        return SparseVector.createSparseVector(10,new int[]{1,3,5,7,9},new double[]{8,6,4,2,0});
    }

    public SparseVector generateSparseD() {
        return SparseVector.createSparseVector(10,new int[]{0,2,4,6,8},new double[]{1,3,5,7,9});
    }

    @Test
    public void euclideanDistance() {
        DenseVector a = generateDenseA();
        DenseVector b = generateDenseB();
        SparseVector c = generateSparseC();
        SparseVector d = generateSparseD();

        // Test the same vector
        assertEquals(0.0,a.euclideanDistance(a),1e-10);
        assertEquals(0.0,b.euclideanDistance(b),1e-10);
        assertEquals(0.0,c.euclideanDistance(c),1e-10);
        assertEquals(0.0,d.euclideanDistance(d),1e-10);

        // Dense-Dense
        assertEquals(18.16590212458495,a.euclideanDistance(b),1e-10);
        assertEquals(18.16590212458495,b.euclideanDistance(a),1e-10);

        // Dense-Sparse
        assertEquals(16.881943016134134,a.euclideanDistance(c),1e-10);
        assertEquals(13.038404810405298,a.euclideanDistance(d),1e-10);
        assertEquals(12.84523257866513,b.euclideanDistance(c),1e-10);
        assertEquals(16.73320053068151,b.euclideanDistance(d),1e-10);

        // Sparse-Dense
        assertEquals(16.881943016134134,c.euclideanDistance(a),1e-10);
        assertEquals(13.038404810405298,d.euclideanDistance(a),1e-10);
        assertEquals(12.84523257866513,c.euclideanDistance(b),1e-10);
        assertEquals(16.73320053068151,d.euclideanDistance(b),1e-10);

        // Sparse-Sparse
        assertEquals(16.881943016134134,c.euclideanDistance(d),1e-10);
        assertEquals(16.881943016134134,d.euclideanDistance(c),1e-10);
    }

    @Test
    public void l1Distance() {
        DenseVector a = generateDenseA();
        DenseVector b = generateDenseB();
        SparseVector c = generateSparseC();
        SparseVector d = generateSparseD();

        // Test the same vector
        assertEquals(0.0,a.l1Distance(a),1e-10);
        assertEquals(0.0,b.l1Distance(b),1e-10);
        assertEquals(0.0,c.l1Distance(c),1e-10);
        assertEquals(0.0,d.l1Distance(d),1e-10);

        // Dense-Dense
        assertEquals(50.0,a.l1Distance(b),1e-10);
        assertEquals(50.0,b.l1Distance(a),1e-10);

        // Dense-Sparse
        assertEquals(45.0,a.l1Distance(c),1e-10);
        assertEquals(30.0,a.l1Distance(d),1e-10);
        assertEquals(25.0,b.l1Distance(c),1e-10);
        assertEquals(44.0,b.l1Distance(d),1e-10);

        // Sparse-Dense
        assertEquals(45.0,c.l1Distance(a),1e-10);
        assertEquals(30.0,d.l1Distance(a),1e-10);
        assertEquals(25.0,c.l1Distance(b),1e-10);
        assertEquals(44.0,d.l1Distance(b),1e-10);

        // Sparse-Sparse
        assertEquals(45.0,c.l1Distance(d),1e-10);
        assertEquals(45.0,d.l1Distance(c),1e-10);
    }

    @Test
    public void cosineDistance() {
        DenseVector a = generateDenseA();
        DenseVector b = generateDenseB();
        SparseVector c = generateSparseC();
        SparseVector d = generateSparseD();

        // Test the same vector
        assertEquals(0.0,a.cosineDistance(a),1e-10);
        assertEquals(0.0,b.cosineDistance(b),1e-10);
        assertEquals(0.0,c.cosineDistance(c),1e-10);
        assertEquals(0.0,d.cosineDistance(d),1e-10);

        // Dense-Dense
        assertEquals(0.5789473684210527,a.cosineDistance(b),1e-10);
        assertEquals(0.5789473684210527,b.cosineDistance(a),1e-10);

        // Dense-Sparse
        assertEquals(0.6755571577384749,a.cosineDistance(c),1e-10);
        assertEquals(0.35439983372499706,a.cosineDistance(d),1e-10);
        assertEquals(0.35111431547694993,b.cosineDistance(c),1e-10);
        assertEquals(0.6080284704758911,b.cosineDistance(d),1e-10);

        // Sparse-Dense
        assertEquals(0.6755571577384749,c.cosineDistance(a),1e-10);
        assertEquals(0.35439983372499706,d.cosineDistance(a),1e-10);
        assertEquals(0.35111431547694993,c.cosineDistance(b),1e-10);
        assertEquals(0.6080284704758911,d.cosineDistance(b),1e-10);

        // Sparse-Sparse
        assertEquals(1.0,c.cosineDistance(d),1e-10);
        assertEquals(1.0,d.cosineDistance(c),1e-10);
    }

}

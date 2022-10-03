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

package org.tribuo.math.util;

import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.SparseVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.test.Helpers.testProtoSerialization;

/**
 *
 */
public class MergerTest {

    public static DenseSparseMatrix generateAB() {
        int[] firstIndices = new int[]     {0,3,5,8, 9,10,14,15,18,19};
        double[] firstValues = new double[]{6,4,0,1,12,56, 9, 9,14,20};
        int[] secondIndices = new int[]     {1, 4,9,10,18};
        double[] secondValues = new double[]{5,-4,1, 8,10};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    public static DenseSparseMatrix generateAABB() {
        int[] firstIndices = new int[]     { 0,3,5,8, 9, 10,14,15,18,19};
        double[] firstValues = new double[]{12,8,0,2,24,112,18,18,28,40};
        int[] secondIndices = new int[]     { 1, 4,9,10,18};
        double[] secondValues = new double[]{10,-8,2,16,20};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    public static DenseSparseMatrix generateA() {
        int[] firstIndices = new int[]     {0,3, 5, 9,10,14,15,18,19};
        double[] firstValues = new double[]{6,4,-1, 3,56,-6, 7,12,20};
        int[] secondIndices = new int[]     {  1,9,18};
        double[] secondValues = new double[]{2.5,1, 4};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    public static DenseSparseMatrix generateB() {
        int[] firstIndices = new int[]     {5,8,9,14,15,18};
        double[] firstValues = new double[]{1,1,9,15, 2, 2};
        int[] secondIndices = new int[]     {  1, 4,10,18};
        double[] secondValues = new double[]{2.5,-4, 8,6};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    public static DenseSparseMatrix generateBB() {
        int[] firstIndices = new int[]     {5,8,9,14,15,18};
        double[] firstValues = new double[]{2,2,18,30, 4, 4};
        int[] secondIndices = new int[]     {  1, 4,10,18};
        double[] secondValues = new double[]{5,-8, 16,12};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    public static DenseSparseMatrix generateZipA() {
        int[] firstIndices = new int[]     {0,2,4,6,8,10,12,14,16,18};
        double[] firstValues = new double[]{1,1,1,1,1, 1, 1, 1, 1, 1};
        int[] secondIndices = new int[]     {1,3,5,7,9,11,13,15,17,19};
        double[] secondValues = new double[]{1,1,1,1,1, 1, 1, 1, 1, 1};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    public static DenseSparseMatrix generateZipB() {
        int[] firstIndices = new int[]     {1,3,5,7,9,11,13,15,17,19};
        double[] firstValues = new double[]{1,1,1,1,1, 1, 1, 1, 1, 1};
        int[] secondIndices = new int[]     {0,2,4,6,8,10,12,14,16,18};
        double[] secondValues = new double[]{1,1,1,1,1, 1, 1, 1, 1, 1};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    public static DenseSparseMatrix generateZip() {
        int[] firstIndices = new int[]     {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
        double[] firstValues = new double[]{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        int[] secondIndices = new int[]     {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
        double[] secondValues = new double[]{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

        SparseVector[] array = new SparseVector[2];

        array[0] = SparseVector.createSparseVector(20,firstIndices,firstValues);
        array[1] = SparseVector.createSparseVector(20,secondIndices,secondValues);

        return DenseSparseMatrix.createFromSparseVectors(array);
    }

    @Test
    public void testMatrixHeapMerger() {
        MatrixHeapMerger merger = new MatrixHeapMerger();
        testMerger(merger);
        testProtoSerialization(merger);
    }

    @Test
    public void testHeapMerger() {
        HeapMerger merger = new HeapMerger();
        testMerger(merger);
        testProtoSerialization(merger);
    }

    public static void testMerger(Merger merger) {
        DenseSparseMatrix[] array = new DenseSparseMatrix[2];
        array[0] = generateA();
        array[1] = generateB();

        DenseSparseMatrix output = generateAB();

        DenseSparseMatrix merged = merger.merge(array);

        assertEquals(output,merged, "Merge A - B unsuccessful");

        array[0] = generateB();

        output = generateBB();

        merged = merger.merge(array);

        assertEquals(output,merged, "Merge B - B unsuccessful");

        array[0] = generateZipA();
        array[1] = generateZipB();

        output = generateZip();

        merged = merger.merge(array);

        assertEquals(output,merged, "Merge zip unsuccessful");

        array = new DenseSparseMatrix[4];

        array[0] = generateA();
        array[1] = generateB();
        array[2] = generateA();
        array[3] = generateB();

        output = generateAABB();

        merged = merger.merge(array);

        assertEquals(output,merged, "Merge A - B - A - B unsuccessful");
    }

}

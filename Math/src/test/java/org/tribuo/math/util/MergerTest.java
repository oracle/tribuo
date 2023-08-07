/*
 * Copyright (c) 2015, 2023, Oracle and/or its affiliates. All rights reserved.
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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.math.protos.MergerProto;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

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

    @ParameterizedTest
    @MethodSource("load431Protobufs")
    public void testProto(String name, Merger actualMerger) throws URISyntaxException, IOException {
        Path mergerPath = Paths.get(MergerTest.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(mergerPath)) {
            MergerProto proto = MergerProto.parseFrom(fis);
            Merger merger = ProtoUtil.deserialize(proto);
            assertEquals(actualMerger, merger);
        }
    }

    private static Stream<Arguments> load431Protobufs() throws URISyntaxException, IOException {
    	return Stream.of(
    		      Arguments.of("heap-merger-431.tribuo", new HeapMerger()),
    		      Arguments.of("matrix-merger-431.tribuo", new MatrixHeapMerger()));
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new HeapMerger(), Paths.get("src","test","resources","org","tribuo","math","util","heap-merger-431.tribuo"));
        Helpers.writeProtobuf(new MatrixHeapMerger(), Paths.get("src","test","resources","org","tribuo","math","util","matrix-merger-431.tribuo"));
    }
}

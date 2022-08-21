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

package org.tribuo.math.la;

import org.junit.jupiter.api.Test;
import org.tribuo.math.protos.TensorProto;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.tribuo.math.la.SparseVectorTest.makeMalformedSparseProto;
import static org.tribuo.test.Helpers.testProtoSerialization;

public class DenseSparseMatrixTest {

    @Test
    public void testCreateIdentity() {
        DenseSparseMatrix identity = DenseSparseMatrix.createIdentity(5);
        assertMatrixEquals(new DenseMatrix(new double[][]{new double[]{1.0, 0.0, 0.0, 0.0, 0.0}, new double[]{0.0, 1.0, 0.0, 0.0, 0.0}, new double[]{0.0, 0.0, 1.0, 0.0, 0.0}, new double[]{0.0, 0.0, 0.0, 1.0, 0.0}, new double[]{0.0, 0.0, 0.0, 0.0, 1.0}}),identity);
        identity = DenseSparseMatrix.createIdentity(1);
        assertMatrixEquals(new DenseMatrix(new double[][]{new double[]{1.0}}),identity);
    }
    
    @Test
    public void testCreateDiagonal() {
        DenseSparseMatrix diagonal = DenseSparseMatrix.createDiagonal(new DenseVector(new double[] {1.0, 2.0}));
        assertMatrixEquals(new DenseMatrix(new double[][]{new double[]{1.0, 0.0}, new double[]{0.0, 2.0}}),diagonal);
        diagonal = DenseSparseMatrix.createDiagonal(new DenseVector(new double[] {1.618033988749894, Math.E, Math.PI}));
        assertMatrixEquals(new DenseMatrix(new double[][]{new double[]{1.618033988749894, 0.0, 0.0}, new double[]{0.0, 2.718281828459045, 0.0}, new double[]{0.0, 0.0, 3.141592653589793}}),diagonal);
    }

    @Test
    public void testGetColumn() {
        DenseSparseMatrix diagonal = DenseSparseMatrix.createDiagonal(new DenseVector(new double[] {1.618033988749894, Math.E, Math.PI}));
        SparseVector column = diagonal.getColumn(1);
        assertEquals(3, column.size());
        assertEquals(0, column.get(0));
        assertEquals(Math.E, column.get(1));
        assertEquals(0, column.get(2));
    }

    @Test
    public void serializationTest() {
        DenseSparseMatrix a = DenseSparseMatrix.createDiagonal(new DenseVector(new double[]{1,2,3,4,5,6}));
        testProtoSerialization(a);

        SparseVector[] vectors = new SparseVector[3];
        vectors[0] = new SparseVector(5, new int[]{1}, new double[]{3});
        vectors[1] = new SparseVector(5, new int[0], new double[0]);
        vectors[2] = new SparseVector(5, new int[]{4}, new double[]{3});
        DenseSparseMatrix missingRow = DenseSparseMatrix.createFromSparseVectors(vectors);
        testProtoSerialization(missingRow);

        vectors = new SparseVector[3];
        vectors[0] = new SparseVector(5, new int[]{1}, new double[]{3});
        vectors[1] = new SparseVector(5, new int[]{4}, new double[]{3});
        vectors[2] = new SparseVector(5, new int[0], new double[0]);
        missingRow = DenseSparseMatrix.createFromSparseVectors(vectors);
        testProtoSerialization(missingRow);

        vectors = new SparseVector[3];
        vectors[0] = new SparseVector(5, new int[0], new double[0]);
        vectors[1] = new SparseVector(5, new int[]{1}, new double[]{3});
        vectors[2] = new SparseVector(5, new int[]{4}, new double[]{3});
        missingRow = DenseSparseMatrix.createFromSparseVectors(vectors);
        testProtoSerialization(missingRow);
    }

    @Test
    public void serializationValidationTest() {
        String className = DenseSparseMatrix.class.getName();
        TensorProto negSize = makeMalformedSparseProto(className,new int[]{-1,1}, 2, new int[]{0,0,1,1}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(negSize);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto negNonZero = makeMalformedSparseProto(className,new int[]{5,4}, -1, new int[]{0,0,3,3}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(negNonZero);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto nonZeroMismatch = makeMalformedSparseProto(className,new int[]{5,4}, 3, new int[]{0,0,3,3}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(nonZeroMismatch);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto invalidIndices = makeMalformedSparseProto(className, new int[]{5,4}, 3, new int[]{0,-1,3,4}, new double[2]);
        try {
            Tensor deser = Tensor.deserialize(invalidIndices);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto valueIndicesMismatch = makeMalformedSparseProto(className, new int[]{5,4}, 2, new int[]{0,3,0,4}, new double[1]);
        try {
            Tensor deser = Tensor.deserialize(valueIndicesMismatch);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }

        TensorProto vector = makeMalformedSparseProto(className, new int[]{5}, 2, new int[]{0,0}, new double[]{1,2});
        try {
            Tensor deser = Tensor.deserialize(vector);
            fail("Should have thrown ISE");
        } catch (IllegalStateException e) {
            //pass
        }
    }

    public static void assertMatrixEquals(Matrix expected, Matrix actual) {
        assertEquals(expected.getDimension1Size(), actual.getDimension1Size(), "dim1 differs");
        assertEquals(expected.getDimension2Size(), actual.getDimension2Size(), "dim2 differs");
        assertArrayEquals(expected.getShape(), actual.getShape(), "shape differs");

        for(int i=0; i<expected.getDimension1Size(); i++) {
            for(int j=0; j<expected.getDimension2Size(); j++) {
                assertEquals(expected.get(i,j), actual.get(i,j), 1e-12, "matrix differs at ("+i+", "+j+")");
            }
        }
    }

}

package org.tribuo.math.la;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

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

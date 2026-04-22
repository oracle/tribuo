/*
 * Copyright (c) 2015, 2026, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.kernel;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.protos.KernelProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;

/**
 * An interface for a Mercer kernel function.
 * <p>
 * It's preferable for kernels to override toString.
 */
public interface Kernel extends Configurable, ProtoSerializable<KernelProto>, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Calculates the similarity between two {@link SGDVector}s.
     * @param first The first SGDVector.
     * @param second The second SGDVector.
     * @return A value between 0 and 1, where 1 is most similar and 0 is least similar.
     */
    public double similarity(SGDVector first, SGDVector second);

    /**
     * Compute the kernel matrix of similarities between all rows of the supplied matrix.
     * @param matrix The data matrix.
     * @return The (symmetric) kernel matrix of similarities.
     */
    public default DenseMatrix computeKernelMatrix(Matrix matrix) {
        int numRows = matrix.getDimension1Size();
        // Square matrix
        DenseMatrix output = new DenseMatrix(numRows, numRows);

        for (int i = 0; i < numRows; i++) {
            SGDVector first = matrix.getRow(i);
            for (int j = i; j < numRows; j++) {
                SGDVector second = matrix.getRow(j);
                double sim = similarity(first,second);
                // Symmetric matrix
                output.set(i,j,sim);
                output.set(j,i,sim);
            }
        }

        return output;
    }

    /**
     * Compute the vector of similarities between the vector and all rows of the supplied matrix.
     * @param vec The vector to compare.
     * @param matrix The matrix.
     * @return The vector of similarities.
     */
    public default DenseVector computeSimilarityVector(SGDVector vec, Matrix matrix) {
        int numRows = matrix.getDimension1Size();
        DenseVector output = new DenseVector(numRows);

        for (int i = 0; i < numRows; i++) {
            SGDVector row = matrix.getRow(i);
            double sim = similarity(vec, row);
            output.set(i, sim);
        }

        return output;
    }

    /**
     * Compute the matrix of similarities between all rows of the supplied matrices.
     * @param firstMatrix The first matrix.
     * @param secondMatrix The second matrix.
     * @return The matrix of similarities.
     */
    public default DenseMatrix computeSimilarityMatrix(Matrix firstMatrix, Matrix secondMatrix) {
        int numRows = firstMatrix.getDimension1Size();
        int numColumns = secondMatrix.getDimension1Size();
        DenseMatrix output = new DenseMatrix(numRows, numColumns);

        for (int i = 0; i < numRows; i++) {
            SGDVector first = firstMatrix.getRow(i);
            for (int j = 0; j < numColumns; j++) {
                SGDVector second = secondMatrix.getRow(j);
                double sim = similarity(first,second);
                output.set(i,j,sim);
            }
        }

        return output;
    }

    /**
     * Deserializes the kernel from the supplied protobuf.
     * @param proto The protobuf to deserialize.
     * @return The kernel.
     */
    public static Kernel deserialize(KernelProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}

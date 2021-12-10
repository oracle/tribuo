/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.onnx;

import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Tribuo Math specific helper functions for building ONNX protos.
 */
public abstract class ONNXMathUtils {

    /**
     * Private constructor for abstract util class.
     */
    private ONNXMathUtils() {}

    /**
     * Builds a {@link ONNXInitializer} containing the {@link SGDVector}.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param vector the SGDVector to store in the proto.
     * @return A TensorProto containing the vector.
     */
    public static ONNXInitializer floatVector(ONNXContext context, String name, SGDVector vector) {
        return context.floatTensor(name, Collections.singletonList(vector.size()),
        (FloatBuffer fb) -> vector.forEach(vt -> fb.put(vt.index,(float) vt.value)));
    }

    /**
     * Builds a {@link ONNXInitializer} containing the {@link Matrix}.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param matrix the matrix to store in the proto.
     * @param transpose Whether to transpose the vector before writing it.
     * @return A TensorProto containing the matrix
     */
    public static ONNXInitializer floatMatrix(ONNXContext context, String name, Matrix matrix, boolean transpose) {
        List<Integer> dims = Arrays.stream(matrix.getShape()).boxed().collect(Collectors.toList());
        if(transpose) {
            Collections.reverse(dims);
        }
        return context.floatTensor(name, dims,
                fb -> matrix.forEach(mt -> {
                    int address = transpose
                            ? mt.j * matrix.getDimension1Size() + mt.i
                            : mt.i * matrix.getDimension2Size() + mt.j;
                    fb.put(address, (float) mt.value);
                }));
    }
}

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

import ai.onnx.proto.OnnxMl;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXUtils;

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
     * Builds a TensorProto containing the {@link SGDVector}.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param vector the SGDVector to store in the proto.
     * @return A TensorProto containing the vector.
     */
    public static OnnxMl.TensorProto floatVectorBuilder(ONNXContext context, String name, SGDVector vector) {
        return ONNXUtils.floatTensorBuilder(context, name, Collections.singletonList(vector.size()),
                (FloatBuffer fb) -> vector.forEach(vt -> fb.put(vt.index,(float) vt.value)));
    }

    /**
     * Builds a TensorProto containing the {@link Matrix}.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param matrix the matrix to store in the proto.
     * @param transpose Whether to transpose the vector before writing it.
     * @return A TensorProto containing the matrix
     */
    public static OnnxMl.TensorProto floatMatrixBuilder(ONNXContext context, String name, Matrix matrix, boolean transpose) {
        List<Integer> dims = Arrays.stream(matrix.getShape()).boxed().collect(Collectors.toList());
        if(transpose) {
            Collections.reverse(dims);
        }
        return ONNXUtils.floatTensorBuilder(context, name,
                dims,
                fb -> matrix.forEach(mt -> {
                    int address = transpose
                            ? mt.j * matrix.getDimension1Size() + mt.i
                            : mt.i * matrix.getDimension2Size() + mt.j;
                    fb.put(address, (float) mt.value);
                }));
    }

}

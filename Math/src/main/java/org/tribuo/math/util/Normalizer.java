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

package org.tribuo.math.util;

import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;

/**
 * Normalizes, but first subtracts the minimum value (to ensure positivity).
 */
public class Normalizer implements VectorNormalizer, Serializable {
    private static final long serialVersionUID = 1L;

    @Override
    public double[] normalize(double[] input) {
        double[] output = Arrays.copyOf(input, input.length);
        normalizeInPlace(output);
        return output;
    }

    @Override
    public void normalizeInPlace(double[] input) {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < input.length; i++) {
            if (min > input[i]) {
                min = input[i];
            }
        }
        min -= 0.01;
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            input[i] -= min;
            sum += input[i];
        }
        for (int i = 0; i < input.length; i++) {
            input[i] /= sum;
        }
    }

    /**
     * Applies ONNX ReduceMin, Sub, ReduceSum, and Div operations to input.
     * @param input The node to be normalized according to this implementation.
     * @return the node representing Div, the final applie operation.
     */
    @Override
    public ONNXNode exportNormalizer(ONNXNode input) {
        ONNXContext onnx = input.onnxContext();
        ONNXInitializer sumAxes = onnx.array("sum_axes", new long[]{1});

        ONNXNode min = input.apply(ONNXOperators.REDUCE_MIN, Collections.singletonMap("axes", new int[]{1}));

        ONNXNode sub = input.apply(ONNXOperators.SUB, min);

        ONNXNode sum = sub.apply(ONNXOperators.REDUCE_SUM, sumAxes);

        return sub.apply(ONNXOperators.DIV, sum);
    }
}

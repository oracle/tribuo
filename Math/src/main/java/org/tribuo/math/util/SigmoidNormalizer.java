/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Normalizes the input by applying a logistic sigmoid to each element.
 * Used when the input is independent dimensions of log space.
 */
public class SigmoidNormalizer implements VectorNormalizer, Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * A logistic sigmoid function.
     * @param input The input to sigmoid.
     * @return The logistic function applied to the input.
     */
    public static double sigmoid(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    @Override
    public double[] normalize(double[] input) {
        double[] output = Arrays.copyOf(input,input.length);
        normalizeInPlace(output);
        return output;
    }

    @Override
    public void normalizeInPlace(double[] input) {
        for (int i = 0; i < input.length; i++) {
            input[i] = sigmoid(input[i]);
        }
    }

    /**
     * Returns the ONNX sigmoid node, operating independently over each element.
     * @param input The node to be normalized according to this implementation.
     * @return the sigmoid node applied to input.
     */
    @Override
    public ONNXNode exportNormalizer(ONNXNode input) {
        return input.apply(ONNXOperators.SIGMOID);
    }
}

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

import ai.onnx.proto.OnnxMl;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXOperators;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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
     * Returns a list of node protos containing a ReduceMin, then a Sub, then a ReduceSum, then a Div.
     * @param context The ONNX context object for name generation.
     * @param input The name of the input to normalize.
     * @param output The name of the normalized output.
     * @return The node protos representing this normalizer.
     */
    @Override
    public List<OnnxMl.NodeProto> exportNormalizer(ONNXContext context, String input, String output) {
        List<OnnxMl.NodeProto> protos = new ArrayList<>();

        String minOutput = context.generateUniqueName("min_output");
        OnnxMl.NodeProto min = ONNXOperators.REDUCE_MIN.build(context,input,minOutput,Collections.singletonMap("axes",new int[]{1}));
        protos.add(min);

        String subOutput = context.generateUniqueName("sub_output");
        OnnxMl.NodeProto sub = ONNXOperators.SUB.build(context,new String[]{input,minOutput},subOutput);
        protos.add(sub);

        String sumOutput = context.generateUniqueName("sum_output");
        OnnxMl.NodeProto sum = ONNXOperators.REDUCE_SUM.build(context,subOutput,sumOutput,Collections.singletonMap("axes",new int[]{1}));
        protos.add(sum);

        OnnxMl.NodeProto div = ONNXOperators.DIV.build(context,new String[]{subOutput,sumOutput},output);
        protos.add(div);

        return protos;
    }
}

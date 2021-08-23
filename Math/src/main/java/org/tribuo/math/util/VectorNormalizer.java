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

import ai.onnx.proto.OnnxMl;
import org.tribuo.onnx.ONNXContext;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * A functional interface that generates a normalized version of a double array.
 */
public interface VectorNormalizer extends Serializable {

    /**
     * Normalizes the input array in some fashion specified by the class.
     * @param input The input to normalize.
     * @return The normalized array.
     */
    public double[] normalize(double[] input);

    /**
     * In place normalization of the array.
     * <p>
     * Note: This default implementation performs a copy by calling the standard normalize method.
     * @param input The input to normalize.
     */
    default public void normalizeInPlace(double[] input) {
        double[] output = normalize(input);
        for (int i = 0; i < input.length; i++) {
            input[i] = output[i];
        }
    }

    /**
     * Exports this normalizer as a list of ONNX NodeProtos.
     * <p>
     * For compatibility reasons the default implementation returns an empty list, and
     * normalizers which use the default implementation will not be able to export
     * their models to ONNX. It is recommended that this method is overridden to
     * support ONNX export, and in a future version of Tribuo this default implementation
     * will be removed.
     * @param context The ONNX context object for name generation.
     * @param input The name of the input to normalize.
     * @param output The name of the normalized output.
     * @return A list of node protos representing the normalization operation.
     */
    default public List<OnnxMl.NodeProto> exportNormalizer(ONNXContext context, String input, String output) {
        Logger.getLogger(this.getClass().getName()).severe("Tried to export a normalizer to ONNX format, but this is not implemented.");
        return Collections.emptyList();
    }

}

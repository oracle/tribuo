/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.math.protos.NormalizerProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXNode;

import java.io.Serializable;
import java.util.logging.Logger;

/**
 * A functional interface that generates a normalized version of a double array.
 */
public interface VectorNormalizer extends ProtoSerializable<NormalizerProto>, Serializable {

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
     * Exports this normalizer to ONNX, returning the leaf of the appended graph
     * and writing the nodes needed for normalization into the {@link ONNXContext}
     * that {@code input} belongs to.
     * <p>
     * For compatibility reasons this method has a default implementation, though
     * when called it will throw an {@code IllegalStateException}. In a future
     * version this method will not have a default implementation and normalizers
     * will be required to provide ONNX support.
     * @param input The node to be normalized according to this implementation.
     * @return The leaf node of the graph of operations added to normalize input.
     */
    default ONNXNode exportNormalizer(ONNXNode input) {
        Logger.getLogger(this.getClass().getName()).severe("Tried to export a normalizer to ONNX format, but this is not implemented.");
        throw new IllegalStateException("Normalizer " + this.getClass() + " cannot be exported in ONNX models.");
    }

    /**
     * Deserializes the vector normalizer from the supplied protobuf.
     * @param proto The protobuf to deserialize.
     * @return The normalizer.
     */
    public static VectorNormalizer deserialize(NormalizerProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}

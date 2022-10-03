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

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import org.tribuo.math.protos.NormalizerProto;
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
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs a Normalizer.
     */
    public SigmoidNormalizer() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static SigmoidNormalizer deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new SigmoidNormalizer();
    }

    @Override
    public NormalizerProto serialize() {
        NormalizerProto.Builder normalizerProto = NormalizerProto.newBuilder();
        normalizerProto.setClassName(this.getClass().getName());
        normalizerProto.setVersion(CURRENT_VERSION);
        return normalizerProto.build();
    }

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

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } else if (o == null || getClass() != o.getClass()) {
            return false;
        } else {
            return true;
        }
    }

    @Override
    public int hashCode() {
        return 31;
    }
}

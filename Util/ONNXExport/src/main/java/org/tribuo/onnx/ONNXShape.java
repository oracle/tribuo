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

package org.tribuo.onnx;

import onnx.OnnxMl;

public final class ONNXShape {

    public final long UNKNOWN_DIMENSION = -1;

    private final long[] dimensions;
    private final String[] dimensionOverrides;

    public ONNXShape(long[] dimensions) {
        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] == UNKNOWN_DIMENSION) {
                throw new IllegalArgumentException("Must specify a name for any unbound dimensions, at idx " + i + " found " + dimensions[i]);
            }
        }
        this.dimensions = dimensions;
        this.dimensionOverrides = null;
    }

    public ONNXShape(long[] dimensions, String[] dimensionOverrides) {
        if (dimensions.length != dimensionOverrides.length) {
            throw new IllegalArgumentException("Must supply dimensions of equal length, found " + dimensions.length + ", and " + dimensionOverrides.length);
        }
        for (int i = 0; i < dimensions.length; i++) {
            if (((dimensions[i] == UNKNOWN_DIMENSION) && (dimensionOverrides[i] == null)) || ((dimensions[i] != UNKNOWN_DIMENSION) && (dimensionOverrides[i] != null))) {
                throw new IllegalArgumentException("Only one of dimensions and dimensionOverrides must contain a value, at idx " + i + " found " + dimensions[i] + " and " + dimensionOverrides[i]);
            }
        }
        this.dimensions = dimensions;
        this.dimensionOverrides = dimensionOverrides;
    }

    public OnnxMl.TensorShapeProto getProto() {
        OnnxMl.TensorShapeProto.Builder builder = OnnxMl.TensorShapeProto.newBuilder();
        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] == -1) {
                builder.addDim(OnnxMl.TensorShapeProto.Dimension.newBuilder().setDimParam(dimensionOverrides[i]).build());
            } else {
                builder.addDim(OnnxMl.TensorShapeProto.Dimension.newBuilder().setDimValue(dimensions[i]).build());
            }
        }
        return builder.build();
    }

}

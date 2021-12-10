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

package org.tribuo.util.onnx;

import ai.onnx.proto.OnnxMl;

/**
 * A subclass of {@link ONNXRef} specialized for {@link ai.onnx.proto.OnnxMl.NodeProto}. It has no
 * specific behavior, for usage see {@link ONNXRef}.
 * <p>
 * N.B. this class should only be instantiated via {@link ONNXContext} or methods on {@link ONNXRef}.
 */
public final class ONNXNode extends ONNXRef<OnnxMl.NodeProto> {
    private final int outputIndex;

    ONNXNode(ONNXContext onnx, OnnxMl.NodeProto backRef, String basename, int outputIndex) {
        super(onnx, backRef, basename);
        this.outputIndex = outputIndex;
    }

    @Override
    public String getReference() {
        return backRef.getOutput(outputIndex);
    }
}

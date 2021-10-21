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

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.util.MutableLong;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Context used while an ONNX model is being generated.
 */
public final class ONNXContext {

    private final Map<String, MutableLong> nameMap;

    private final OnnxMl.GraphProto.Builder protoBuilder;

    /**
     * Creates an empty ONNX context.
     */
    public ONNXContext() {
        this.nameMap = new HashMap<>();
        this.protoBuilder = OnnxMl.GraphProto.newBuilder();
    }

    /**
     * Generates a unique name by appending the counter for that name.
     * @param name The name.
     * @return A unique version of that name.
     */
    public String generateUniqueName(String name) {
        MutableLong counter = nameMap.computeIfAbsent(name,k -> new MutableLong());
        String newName = name + "_" + counter.longValue();
        counter.increment();
        return newName;
    }

    public void addInitializer(OnnxMl.TensorProto tensor) {
        protoBuilder.addInitializer(tensor);
    }

    public void addNode(OnnxMl.NodeProto node) {
        protoBuilder.addNode(node);
    }

    public void addAllNodes(List<OnnxMl.NodeProto> nodes) {
        protoBuilder.addAllNode(nodes);
    }

    public void setName(String name) {
        protoBuilder.setName(name);
    }

    public void addInput(OnnxMl.ValueInfoProto input) {
        protoBuilder.addInput(input);
    }

    public void addOutput(OnnxMl.ValueInfoProto output) {
        protoBuilder.addOutput(output);
    }

    public String getOutputName(int index) {
        return protoBuilder.getOutput(index).getName();
    }

    public OnnxMl.GraphProto buildGraph() {
        return protoBuilder.build();
    }

    public String getInputName(int index) {
        return protoBuilder.getInput(index).getName();
    }
}

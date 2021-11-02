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

    /**
     * Adds an initializer to the graph.
     * @param tensor The initializer to add.
     */
    public void addInitializer(OnnxMl.TensorProto tensor) {
        protoBuilder.addInitializer(tensor);
    }

    /**
     * Adds a node to the graph.
     * @param node The node to add.
     */
    public void addNode(OnnxMl.NodeProto node) {
        protoBuilder.addNode(node);
    }

    /**
     * Adds all the nodes to the graph.
     * @param nodes The nodes to add.
     */
    public void addAllNodes(List<OnnxMl.NodeProto> nodes) {
        protoBuilder.addAllNode(nodes);
    }

    /**
     * Sets the graph name.
     * @param name The graph name.
     */
    public void setName(String name) {
        protoBuilder.setName(name);
    }

    /**
     * Adds an input to the graph.
     * @param input The input to add.
     */
    public void addInput(OnnxMl.ValueInfoProto input) {
        protoBuilder.addInput(input);
    }

    /**
     * Adds an output to the graph.
     * @param output The output to add.
     */
    public void addOutput(OnnxMl.ValueInfoProto output) {
        protoBuilder.addOutput(output);
    }

    /**
     * Gets the output name at the specified index.
     * <p>
     * Throws {@link IndexOutOfBoundsException} if the index is out of bounds.
     * @param index The output index.
     * @return The output name.
     */
    public String getOutputName(int index) {
        if (index < 0 || protoBuilder.getOutputCount() < index) {
            throw new IndexOutOfBoundsException("Invalid index, expected [0," + protoBuilder.getOutputCount() + "), received " + index);
        }
        return protoBuilder.getOutput(index).getName();
    }

    /**
     * Gets the input name at the specified index.
     * <p>
     * Throws {@link IndexOutOfBoundsException} if the index is out of bounds.
     * @param index The input index.
     * @return The input name.
     */
    public String getInputName(int index) {
        if (index < 0 || protoBuilder.getOutputCount() < index) {
            throw new IndexOutOfBoundsException("Invalid index, expected [0," + protoBuilder.getOutputCount() + "), received " + index);
        }
        return protoBuilder.getInput(index).getName();
    }

    /**
     * Builds the graph contained in this context.
     * @return The graph.
     */
    public OnnxMl.GraphProto buildGraph() {
        return protoBuilder.build();
    }

}

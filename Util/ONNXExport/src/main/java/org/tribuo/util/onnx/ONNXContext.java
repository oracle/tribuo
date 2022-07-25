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

import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Context object used to scope and manage the creation of ONNX {@link ai.onnx.proto.OnnxMl.GraphProto} and {@link ai.onnx.proto.OnnxMl.ModelProto}
 * instances. A single instance of ONNXContext should be used to create an ONNX graph/model, and mixing instances of
 * ONNXContext or of {@link ONNXRef}s produced by multiple ONNXContexts is not supported.
 * <p>
 * The ONNXContext has all of the logic needed to produce ONNX graphs, but is typically used explicitly to produce leaf
 * nodes of graphs (inputs, outputs, and weight matrices) that have more fluent interfaces to {@link ONNXContext#operation(ONNXOperator, List, List, Map)}.
 * Produced ONNX protobuf objects are encapsulated by instances of {@link ONNXRef} and its subclasses.
 */
public final class ONNXContext {

    private final Map<String, Long> nameMap;

    private final OnnxMl.GraphProto.Builder protoBuilder;

    /**
     * Creates an empty ONNX context.
     */
    public ONNXContext() {
        this.nameMap = new HashMap<>();
        this.protoBuilder = OnnxMl.GraphProto.newBuilder();
    }

    /**
     * Base method for creating {@link ONNXNode}s from {@link ONNXOperator} and inputs. Returns an instance of ONNXNode
     * for each output of the ONNXOperator. The graph elements created by the operation are added to the calling
     * ONNXContext instance. All inputs must belong to the calling instance of ONNXContext. This is the root method for
     * constructing ONNXNodes which all other methods on ONNXContext and {@code ONNXRef} call.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param inputs A list of {@link ONNXRef}s created by this instance of ONNXContext.
     * @param outputs A list of names that the output nodes of {@code op} should take.
     * @param attributes A map of attributes of the operation, passed to {@link ONNXOperator#build(ONNXContext, String, String, Map)}.
     * @param <T> The ONNXRef type of inputs
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public <T extends ONNXRef<?>> List<ONNXNode> operation(ONNXOperator op,
                                                    List<T> inputs,
                                                    List<String> outputs,
                                                    Map<String, Object> attributes) {
        if(!inputs.stream().allMatch(n -> n.context == this)) {
            throw new IllegalArgumentException("All input nodes must belong to this ONNXContext");
        }
        OnnxMl.NodeProto opProto = op.build(this,
                inputs.stream().map(ONNXRef::getReference).toArray(String[]::new),
                outputs.stream().map(this::generateUniqueName).toArray(String[]::new),
                attributes);
        protoBuilder.addNode(opProto);
        return IntStream.range(0, outputs.size()).mapToObj(i-> new ONNXNode(this, opProto, outputs.get(i), i)).collect(Collectors.toList());
    }

    /**
     * Method for creating {@link ONNXNode}s from {@link ONNXOperator} and inputs. Returns a single ONNXNode and throws
     * IllegalStateException if the operator has multiple outputs. The graph elements created by the operation are added
     * to the calling ONNXContext instance. All inputs must belong to the calling instance of ONNXContext.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param inputs A list of {@link ONNXRef}s created by this instance of ONNXContext.
     * @param outputName Name that the output node of {@code op} should take.
     * @param attributes A map of attributes of the operation, passed to {@link ONNXOperator#build(ONNXContext, String, String, Map)}.
     * @param <T> The ONNXRef type of inputs
     * @return An {@link ONNXNode} that is the output nodes of {@code op}.
     */
    public <T extends ONNXRef<?>> ONNXNode operation(ONNXOperator op, List<T> inputs, String outputName, Map<String, Object> attributes) {
        List<ONNXNode> opOutputs = operation(op, inputs, Collections.singletonList(outputName), attributes);
        if(opOutputs.get(0).backRef.getOutputList().size() > 1) {
            throw new IllegalStateException("Requested a single output from operation " + op.getOpName() + " which produced " + opOutputs.get(0).backRef.getOutputList().size() + " outputs");
        } else {
            return opOutputs.get(0);
        }
    }

    /**
     * Method for creating {@link ONNXNode}s from {@link ONNXOperator} instances and inputs. Returns a single ONNXNode
     * and throws IllegalStateException if the operator has multiple outputs. The graph elements created by the
     * operation are added to the calling ONNXContext instance. All inputs must belong to the calling instance of
     * ONNXContext.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param inputs A list of {@link ONNXRef}s created by this instance of ONNXContext.
     * @param outputName Name that the output node of {@code op} should take.
     * @param <T> The ONNXRef type of inputs
     * @return An {@link ONNXNode} that is the output nodes of {@code op}.
     */
    public <T extends ONNXRef<?>> ONNXNode operation(ONNXOperator op, List<T> inputs, String outputName) {
        return operation(op, inputs, outputName, Collections.emptyMap());
    }

    /**
     * Creates an {@link ONNXOperators#IDENTITY} node connecting {@code input} to {@code output}, effectively permitting
     * assignment of values.
     * @param input The input node / right-hand side of the assignment.
     * @param output The output node / left-hand side of the assignment.
     * @param <LHS> the {@link ONNXRef} type of the output.
     * @param <RHS> the {@link ONNXRef} type of the input.
     * @return the output node that was assigned to.
     */
    public <LHS extends ONNXRef<?>, RHS extends ONNXRef<?>> LHS assignTo(RHS input, LHS output) {
        if(!(input.context == output.context && input.context == this)) {
            throw new IllegalArgumentException("both input and output must both belong to this ONNXContext");
        }
        OnnxMl.NodeProto idNode = ONNXOperators.IDENTITY.build(this, input.getReference(), output.getReference());
        protoBuilder.addNode(idNode);
        return output;
    }

    /**
     * Creates an input node for this ONNXContext, with the given name, of dimension
     * [batch_size, {@code featureDimension}], and of type float32.
     * @param name The name for this input node.
     * @param featureDimension the second dimension of this input node.
     * @return An {@link ONNXPlaceholder} instance representing this input node.
     */
    public ONNXPlaceholder floatInput(String name, int featureDimension) {
        OnnxMl.TypeProto inputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1, featureDimension}, new String[]{"batch", null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValue = OnnxMl.ValueInfoProto.newBuilder()
                .setType(inputType)
                .setName(name).build();
        protoBuilder.addInput(inputValue);
        return new ONNXPlaceholder(this, inputValue, name);
    }

    /**
     * Creates an input node for this ONNXContext, with the name "input", of dimension
     * [batch_size, {@code featureDimension}], and of type float32.
     * @param featureDimension the second dimension of this input node.
     * @return An {@link ONNXPlaceholder} instance representing this input node.
     */
    public ONNXPlaceholder floatInput(int featureDimension) {
        return floatInput("input", featureDimension);
    }

    /**
     * Creates an output node for this ONNXContext, with the given name, of dimension
     * [batch_size, {@code outputDimension}], and of type float32.
     * @param name the name for this output node.
     * @param outputDimension The second dimension of this output node.
     * @return An {@link ONNXPlaceholder} instance representing this output node.
     */
    public ONNXPlaceholder floatOutput(String name, int outputDimension) {
        OnnxMl.TypeProto outputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputDimension}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder()
                .setType(outputType)
                .setName(name).build();
        protoBuilder.addOutput(outputValueProto);
        return new ONNXPlaceholder(this, outputValueProto, name);
    }

    /**
     * Creates an output node for this ONNXContext, with the name "output", of dimension
     * [batch_size, {@code outputDimension}], and of type float32.
     * @param outputDimension The second dimension of this output node.
     * @return An {@link ONNXPlaceholder} instance representing this output node.
     */
    public ONNXPlaceholder floatOutput(int outputDimension) {
        return floatOutput("output", outputDimension);
    }

    /**
     * Creates a tensor for this ONNXContext, populated as {@link ONNXUtils#floatTensorBuilder(ONNXContext, String, List, Consumer)}.
     * @param baseName The name for this tensor in the ONNX graph.
     * @param dims The dimensions of this tensor.
     * @param populate A function populating the {@link FloatBuffer} that backs this tensor.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer floatTensor(String baseName, List<Integer> dims, Consumer<FloatBuffer> populate) {
        OnnxMl.TensorProto tens = ONNXUtils.floatTensorBuilder(this, baseName, dims, populate);
        protoBuilder.addInitializer(tens);
        return new ONNXInitializer(this, tens, baseName);
    }

    /**
     * Creates a long tensor for this ONNXContext, populated according to parameters.
     * @param baseName The name for this tensor in the ONNX graph.
     * @param parameters The long[] to populate the tensor.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer array(String baseName, long[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        protoBuilder.addInitializer(tens);
        return new ONNXInitializer(this, tens, baseName);
    }

    /**
     * Creates an int tensor for this ONNXContext, populated according to parameters.
     * @param baseName The name for this tensor in the ONNX graph.
     * @param parameters The int[] to populate the tensor.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer array(String baseName, int[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        protoBuilder.addInitializer(tens);
        return new ONNXInitializer(this, tens, baseName);
    }

    /**
     * Creates a float tensor for this ONNXContext, populated according to parameters.
     * @param baseName The name for this tensor in the ONNX graph.
     * @param parameters The float[] to populate the tensor.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer array(String baseName, float[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        protoBuilder.addInitializer(tens);
        return new ONNXInitializer(this, tens, baseName);
    }

    /**
     * Creates a tensor for this ONNXContext, populated according to parameters.
     * @param baseName The name for this tensor in the ONNX graph.
     * @param parameters The double[] to populate the tensor.
     * @param downcast Whether to downcast {@code parameters} to float32 in the ONNX graph.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer array(String baseName, double[] parameters, boolean downcast) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters, downcast);
        protoBuilder.addInitializer(tens);
        return new ONNXInitializer(this, tens, baseName);
    }

    /**
     * Creates a float tensor for this ONNXContext, populated according to parameters.
     * <p>
     * As with {@link ONNXUtils#arrayBuilder(ONNXContext, String, double[], boolean)} the doubles will be downcast to
     * float32.
     * @param baseName The name for this tensor in the ONNX graph.
     * @param parameters The double[] to populate the tensor.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer array(String baseName, double[] parameters) {
        return array(baseName, parameters, true);
    }

    /**
     * Creates a float scalar constant for this ONNXContext.
     * @param baseName The name for this constant in the ONNX graph.
     * @param value The float to populate the constant.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer constant(String baseName, float value) {
        OnnxMl.TensorProto constant = OnnxMl.TensorProto.newBuilder()
                .setName(generateUniqueName(baseName))
                .setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber())
                .addFloatData(value)
                .build();
        protoBuilder.addInitializer(constant);
        return new ONNXInitializer(this, constant, baseName);
    }

    /**
     * Creates a long scalar constant for this ONNXContext.
     * @param baseName The name for this constant in the ONNX graph.
     * @param value The long to populate the constant.
     * @return An {@link ONNXInitializer} instance representing this tensor.
     */
    public ONNXInitializer constant(String baseName, long value) {
        OnnxMl.TensorProto constant = OnnxMl.TensorProto.newBuilder()
                .setName(generateUniqueName(baseName))
                .setDataType(OnnxMl.TensorProto.DataType.INT64.getNumber())
                .addInt64Data(value)
                .build();
        protoBuilder.addInitializer(constant);
        return new ONNXInitializer(this, constant, baseName);
    }

    /**
     * Generates a unique name by appending the counter for that name.
     * @param name The name.
     * @return A unique version of that name.
     */
    String generateUniqueName(String name) {
        long counter = nameMap.computeIfAbsent(name,k -> 0L);
        String newName = name + "_" + counter;
        nameMap.put(name,counter + 1);
        return newName;
    }

    /**
     * Sets the graph name.
     * @param name The graph name.
     */
    public void setName(String name) {
        protoBuilder.setName(name);
    }

    /**
     * Builds the ONNX graph represented by this context.
     * @return The ONNX graph proto.
     */
    public OnnxMl.GraphProto buildGraph() {
        return protoBuilder.build();
    }
}

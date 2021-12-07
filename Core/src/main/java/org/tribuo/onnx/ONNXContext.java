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
import com.google.protobuf.GeneratedMessageV3;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.Tribuo;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Context used while an ONNX model is being generated.
 */
public final class ONNXContext {

    public abstract class ONNXRef<T extends GeneratedMessageV3> {
        // Unfortunately there is no other shared supertype for OnnxML protobufs
        protected T backRef;
        private final String baseName;


        ONNXRef(T backRef, String baseName) {
            this.backRef = backRef;
            this.baseName = baseName;
        }

        public abstract String getReference();

        public String getBaseName() {
            return baseName;
        }

        public ONNXContext onnx() {
            return ONNXContext.this;
        }


        public List<ONNXNode> apply(ONNXOperators op, List<ONNXRef<?>> otherInputs, List<String> outputs, Map<String, Object> attributes) {
            List<ONNXRef<?>> allInputs = new ArrayList<>();
            allInputs.add(this);
            allInputs.addAll(otherInputs);
            return ONNXContext.this.operation(op, allInputs, outputs, attributes);
        }

        public List<ONNXNode> apply(ONNXOperators op, List<String> outputs, Map<String, Object> attributes) {
            return ONNXContext.this.operation(op, Collections.singletonList(this), outputs, attributes);
        }

        public ONNXNode apply(ONNXOperators op) {
            return ONNXContext.this.operation(op, Collections.singletonList(this), getBaseName() + "_" + op.opName, Collections.emptyMap());
        }

        public ONNXNode apply(ONNXOperators op, Map<String, Object> attributes) {
            return ONNXContext.this.operation(op, Collections.singletonList(this), getBaseName() + "_" + op.opName, attributes);
        }

        public ONNXNode apply(ONNXOperators op, ONNXRef<?> other, Map<String, Object> attributes) {
            return ONNXContext.this.operation(op, Arrays.asList(this, other), getBaseName() + "_" + op.opName + "_" + other.getBaseName(), attributes);
        }

        public ONNXNode apply(ONNXOperators op, ONNXRef<?> other) {
            return ONNXContext.this.operation(op, Arrays.asList(this, other), getBaseName() + "_" + op.opName + "_" + other.getBaseName(), Collections.emptyMap());
        }

        public ONNXNode apply(ONNXOperators op, List<ONNXRef<?>> others) {
            return apply(op, others, Collections.singletonList(getBaseName() + "_" + others.stream().map(ONNXRef::getBaseName).collect(Collectors.joining("_"))), Collections.emptyMap()).get(0);
        }

        public ONNXNode apply(ONNXOperators op, List<ONNXRef<?>> others, String outputName) {
            return apply(op, others, Collections.singletonList(outputName), Collections.emptyMap()).get(0);
        }

        public <T extends ONNXRef<?>> T assignTo(T output) {
            OnnxMl.NodeProto idNode = ONNXOperators.IDENTITY.build(ONNXContext.this, this.getReference(), output.getReference());
            ONNXContext.this.addNode(idNode);
            return output;

        }

        public ONNXNode cast(Class<?> clazz) {
            if(clazz.isAssignableFrom(float.class)) {
                return this.apply(ONNXOperators.CAST, Collections.singletonMap("to", OnnxMl.TensorProto.DataType.FLOAT.getNumber()));
            } else {
                throw new IllegalArgumentException("unsupported class for casting: " +clazz.getName());
            }
        }
    }

    public class ONNXNode extends ONNXRef<OnnxMl.NodeProto> {
        //protected OnnxMl.NodeProto backRef;
        private final int outputIndex;

        ONNXNode(OnnxMl.NodeProto backRef, String basename) {
            this(backRef, basename, 0);
        }

        ONNXNode(OnnxMl.NodeProto backRef, String basename, int outputIndex) {
            super(backRef, basename);
            this.outputIndex = outputIndex;
        }

        @Override
        public String getReference() {
            return backRef.getOutput(outputIndex);
        }
    }

    public class ONNXPlaceholder extends ONNXRef<OnnxMl.ValueInfoProto> {
        //protected OnnxMl.ValueInfoProto backRef;

        ONNXPlaceholder(OnnxMl.ValueInfoProto backRef, String basename) {
            super(backRef, basename);
        }

        @Override
        public String getReference() {
            return backRef.getName();
        }
    }

    public class ONNXTensor extends ONNXRef<OnnxMl.TensorProto> {
        //protected OnnxMl.TensorProto backRef;
        ONNXTensor(OnnxMl.TensorProto backRef, String baseName) {
            super(backRef, baseName);
        }

        @Override
        public String getReference() {
            return backRef.getName();
        }
    }

    private final Map<String, MutableLong> nameMap;

    private final OnnxMl.GraphProto.Builder protoBuilder;

    /**
     * Creates an empty ONNX context.
     */
    public ONNXContext() {
        this.nameMap = new HashMap<>();
        this.protoBuilder = OnnxMl.GraphProto.newBuilder();
    }


    <T extends ONNXRef<?>> List<ONNXContext.ONNXNode> operation(ONNXOperators op,
                             List<T> inputs,
                             List<String> outputs,
                             Map<String, Object> attributes) {
        OnnxMl.NodeProto opProto = op.build(this,
                inputs.stream().map(ONNXRef::getReference).toArray(String[]::new),
                outputs.stream().map(this::generateUniqueName).toArray(String[]::new),
                attributes);
        this.addNode(opProto);
        return IntStream.range(0, outputs.size()).mapToObj(i-> new ONNXNode(opProto, outputs.get(i), i)).collect(Collectors.toList());
    }

    public <T extends ONNXRef<?>> ONNXNode operation(ONNXOperators op, List<T> inputs, String outputName, Map<String, Object> attributes) {
        List<ONNXNode> opOutputs = operation(op, inputs, Collections.singletonList(outputName), attributes);
        if(opOutputs.get(0).backRef.getOutputList().size() > 1) {
            throw new IllegalStateException("Requested a single output from operation " + op.opName + " which produced " + opOutputs.get(0).backRef.getOutputList().size() + " outputs");
        } else {
            return opOutputs.get(0);
        }
    }

    public <T extends ONNXRef<?>> ONNXNode operation(ONNXOperators op, List<T> inputs, String outputName) {
        return operation(op, inputs, outputName, Collections.emptyMap());
    }

    ONNXNode operation(ONNXOperators op, List<ONNXRef<?>> inputs, ONNXPlaceholder output, Map<String, Object> attributes) {
        return operation(op, inputs, output.getReference(), attributes);
    }

    private static <T extends GeneratedMessageV3> Optional<ONNXRef> findInGraph(OnnxMl.GraphProtoOrBuilder graph, Function<OnnxMl.GraphProtoOrBuilder, List<T>> getItems, Function<T, String> getName, BiFunction<T, String, ONNXRef> makeRef, String name) {
        return getItems.apply(graph)
                .stream()
                .filter(i -> getName.apply(i).equals(name))
                .map(i -> makeRef.apply(i, name))
                .findFirst();
    }

    public ONNXPlaceholder floatInput(String name, int featureDimension) {
        OnnxMl.TypeProto inputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1, featureDimension}, new String[]{"batch", null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValue = OnnxMl.ValueInfoProto.newBuilder()
                .setType(inputType)
                .setName(name).build();
        protoBuilder.addInput(inputValue);
        return new ONNXPlaceholder(inputValue, name);
    }

    public ONNXPlaceholder floatInput(int featureDimension) {
        return floatInput("input", featureDimension);
    }

    public ONNXPlaceholder floatOutput(String name, int outputDimension) {
        OnnxMl.TypeProto outputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputDimension}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder()
                .setType(outputType)
                .setName(name).build();
        protoBuilder.addOutput(outputValueProto);
        return new ONNXPlaceholder(outputValueProto, name);
    }

    public ONNXPlaceholder floatOutput(int outputDimension) {
        return floatOutput("output", outputDimension);
    }

    public ONNXTensor floatTensor(String baseName, List<Integer> dims, Consumer<FloatBuffer> populate) {
        OnnxMl.TensorProto tens = ONNXUtils.floatTensorBuilder(this, baseName, dims, populate);
        this.addInitializer(tens);
        return new ONNXTensor(tens, baseName);
    }

    public ONNXTensor array(String baseName, long[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        this.addInitializer(tens);
        return new ONNXTensor(tens, baseName);
    }

    public ONNXTensor array(String baseName, int[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        this.addInitializer(tens);
        return new ONNXTensor(tens, baseName);
    }

    public ONNXTensor array(String baseName, float[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        this.addInitializer(tens);
        return new ONNXTensor(tens, baseName);
    }

    public ONNXTensor array(String baseName, double[] parameters, boolean downcast) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters, downcast);
        this.addInitializer(tens);
        return new ONNXTensor(tens, baseName);
    }

    public ONNXTensor array(String baseName, double[] parameters) {
        return array(baseName, parameters, true);
    }

    public ONNXTensor constant(String baseName, float value) {
        OnnxMl.TensorProto constant = OnnxMl.TensorProto.newBuilder()
                .setName(generateUniqueName(baseName))
                .setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber())
                .addFloatData(value)
                .build();
        this.addInitializer(constant);
        return new ONNXTensor(constant, baseName);
    }

    public ONNXTensor constant(String baseName, long value) {
        OnnxMl.TensorProto constant = OnnxMl.TensorProto.newBuilder()
                .setName(generateUniqueName(baseName))
                .setDataType(OnnxMl.TensorProto.DataType.INT64.getNumber())
                .addInt64Data(value)
                .build();
        this.addInitializer(constant);
        return new ONNXTensor(constant, baseName);
    }

    public <O extends ObjectProvenance, M extends Provenancable<O>> OnnxMl.ModelProto model(String domain, long modelVersion, M model) {
        return OnnxMl.ModelProto.newBuilder()
                .setGraph(buildGraph())
                .setDomain(domain)
                .setProducerName("Tribuo")
                .setProducerVersion(Tribuo.VERSION)
                .setModelVersion(modelVersion)
                .addOpsetImport(ONNXOperators.getOpsetProto())
                .setIrVersion(6)
                .setDocString(model.toString())
                .addMetadataProps(OnnxMl.StringStringEntryProto
                        .newBuilder()
                        .setKey(ONNXExportable.PROVENANCE_METADATA_FIELD)
                        .setValue(ONNXExportable.SERIALIZER.marshalAndSerialize(model.getProvenance()))
                        .build())
                .build();
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
        System.out.println("Via old API adding output of name " + output.getName() + " to graph " + protoBuilder.getName());
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

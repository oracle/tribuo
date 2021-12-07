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
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.Tribuo;

import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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


    <T extends ONNXRef<?>> List<ONNXNode> operation(ONNXOperators op,
                                                    List<T> inputs,
                                                    List<String> outputs,
                                                    Map<String, Object> attributes) {
        if(inputs.stream().map(n -> n.onnx).collect(Collectors.toSet()).size() > 1) {
            throw new IllegalArgumentException("All input nodes must belong to the same ONNXContext");
        }
        OnnxMl.NodeProto opProto = op.build(this,
                inputs.stream().map(ONNXRef::getReference).toArray(String[]::new),
                outputs.stream().map(this::generateUniqueName).toArray(String[]::new),
                attributes);
        protoBuilder.addNode(opProto);
        return IntStream.range(0, outputs.size()).mapToObj(i-> new ONNXNode(this, opProto, outputs.get(i), i)).collect(Collectors.toList());
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

    public <LHS extends ONNXRef<?>, RHS extends ONNXRef<?>> RHS assignTo(LHS input, RHS output) {
        if(input.onnx != output.onnx) {
            throw new IllegalArgumentException("All input nodes must belong to the same ONNXContext");
        }
        OnnxMl.NodeProto idNode = ONNXOperators.IDENTITY.build(this, input.getReference(), output.getReference());
        ONNXContext.this.protoBuilder.addNode(idNode);
        return output;
    }

    public ONNXPlaceholder floatInput(String name, int featureDimension) {
        OnnxMl.TypeProto inputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1, featureDimension}, new String[]{"batch", null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValue = OnnxMl.ValueInfoProto.newBuilder()
                .setType(inputType)
                .setName(name).build();
        protoBuilder.addInput(inputValue);
        return new ONNXPlaceholder(this, inputValue, name);
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
        return new ONNXPlaceholder(this, outputValueProto, name);
    }

    public ONNXPlaceholder floatOutput(int outputDimension) {
        return floatOutput("output", outputDimension);
    }

    public ONNXTensor floatTensor(String baseName, List<Integer> dims, Consumer<FloatBuffer> populate) {
        OnnxMl.TensorProto tens = ONNXUtils.floatTensorBuilder(this, baseName, dims, populate);
        protoBuilder.addInitializer(tens);
        return new ONNXTensor(this, tens, baseName);
    }

    public ONNXTensor array(String baseName, long[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        protoBuilder.addInitializer(tens);
        return new ONNXTensor(this, tens, baseName);
    }

    public ONNXTensor array(String baseName, int[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        protoBuilder.addInitializer(tens);
        return new ONNXTensor(this, tens, baseName);
    }

    public ONNXTensor array(String baseName, float[] parameters) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters);
        protoBuilder.addInitializer(tens);
        return new ONNXTensor(this, tens, baseName);
    }

    public ONNXTensor array(String baseName, double[] parameters, boolean downcast) {
        OnnxMl.TensorProto tens = ONNXUtils.arrayBuilder(this, baseName, parameters, downcast);
        protoBuilder.addInitializer(tens);
        return new ONNXTensor(this, tens, baseName);
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
        protoBuilder.addInitializer(constant);
        return new ONNXTensor(this, constant, baseName);
    }

    public ONNXTensor constant(String baseName, long value) {
        OnnxMl.TensorProto constant = OnnxMl.TensorProto.newBuilder()
                .setName(generateUniqueName(baseName))
                .setDataType(OnnxMl.TensorProto.DataType.INT64.getNumber())
                .addInt64Data(value)
                .build();
        protoBuilder.addInitializer(constant);
        return new ONNXTensor(this, constant, baseName);
    }

    public <O extends ObjectProvenance, M extends Provenancable<O>> OnnxMl.ModelProto model(String domain, long modelVersion, M model) {
        return OnnxMl.ModelProto.newBuilder()
                .setGraph(protoBuilder.build())
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
    String generateUniqueName(String name) {
        MutableLong counter = nameMap.computeIfAbsent(name,k -> new MutableLong());
        String newName = name + "_" + counter.longValue();
        counter.increment();
        return newName;
    }

    /**
     * Sets the graph name.
     * @param name The graph name.
     */
    public void setName(String name) {
        protoBuilder.setName(name);
    }
}

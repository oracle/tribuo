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
import com.google.protobuf.GeneratedMessageV3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An abstract reference that represents both a node in an ONNX computation graph and a container for a specific ONNX
 * proto object that denotes that node. In its role as the former it provides a fluent interface for applying
 * {@link ONNXOperator}s to {@link ONNXRef} instances. ONNXRef instances are ultimately created by an {@link ONNXContext}
 * instance, and ONNXRefs created by different instances of ONNXContext are incompatible. All ONNX proto objects
 * produced by calling {@code apply} methods on ONNXRefs are added to a {@link ai.onnx.proto.OnnxMl.GraphProto} field
 * in their governing ONNXContext. Instances of ONNXRef have a backreference to the ONNXContext that created them and
 * can thus be passed around without needing to pass their governing context as well.
 * <p>
 * N.B. This class will be sealed once the library is updated past Java 8. Users should not subclass this class.
 * @param <T> The protobuf type this reference generates.
 */
public abstract class ONNXRef<T extends GeneratedMessageV3> {
    // Unfortunately there is no other shared supertype for OnnxML protobufs
    /**
     * Protobuf reference.
     */
    protected final T backRef;
    private final String baseName;
    /**
     * The ONNX context.
     */
    protected final ONNXContext context;

    /**
     * Creates an ONNXRef for the specified context, protobuf and name.
     * @param context The ONNXContext we're operating in.
     * @param backRef The protobuf reference.
     * @param baseName The name of this reference.
     */
    ONNXRef(ONNXContext context, T backRef, String baseName) {
        this.context = context;
        this.backRef = backRef;
        this.baseName = baseName;
    }

    /**
     * Gets the output name of this object.
     * @return The output name.
     */
    public abstract String getReference();

    /**
     * The name of this object.
     * @return The name.
     */
    public String getBaseName() {
        return baseName;
    }

    /**
     * The context this reference operates in.
     * @return The context.
     */
    public ONNXContext onnxContext() {
        return context;
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, List, Map)}, using this ONNXRef
     * as the first argument to {@code inputs}, with {@code otherInputs} append as subsequent arguments. The other
     * arguments behave as in the analogous method on ONNXContext.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param otherInputs A list of {@link ONNXRef}s created by this instance of ONNXContext.
     * @param outputs A list of names that the output nodes of {@code op} should take.
     * @param attributes A map of attributes of the operation, passed to {@link ONNXOperator#build(ONNXContext, String, String, Map)}.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public List<ONNXNode> apply(ONNXOperator op, List<ONNXRef<?>> otherInputs, List<String> outputs, Map<String, Object> attributes) {
        List<ONNXRef<?>> allInputs = new ArrayList<>();
        allInputs.add(this);
        allInputs.addAll(otherInputs);
        return context.operation(op, allInputs, outputs, attributes);
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, List, Map)}, using this ONNXRef
     * as the argument to {@code inputs}. The other arguments behave as in the analogous method on ONNXContext.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param outputs A list of names that the output nodes of {@code op} should take.
     * @param attributes A map of attributes of the operation, passed to {@link ONNXOperator#build(ONNXContext, String, String, Map)}.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public List<ONNXNode> apply(ONNXOperator op, List<String> outputs, Map<String, Object> attributes) {
        return context.operation(op, Collections.singletonList(this), outputs, attributes);
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String)}, using this ONNXRef
     * as the argument to {@code inputs}. Output names are generated based on the {@link ONNXOperator#getOpName} and the
     * name of the input nodes.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op) {
        return context.operation(op, Collections.singletonList(this), getBaseName() + "_" + op.getOpName(), Collections.emptyMap());
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String)}, using this ONNXRef
     * as the argument to {@code inputs}.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param outputName A name that the output node of {@code op} will take.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op, String outputName) {
        return context.operation(op, Collections.singletonList(this), outputName, Collections.emptyMap());
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String, Map)}, using this ONNXRef
     * as the argument to {@code inputs}. Output names are generated based on the {@link ONNXOperator#getOpName} and the
     * name of the input nodes.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param attributes A map of attributes of the operation, passed to {@link ONNXOperator#build(ONNXContext, String, String, Map)}.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op, Map<String, Object> attributes) {
        return context.operation(op, Collections.singletonList(this), getBaseName() + "_" + op.getOpName(), attributes);
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String, Map)}, passing this ONNXRef
     * and {@code other} as a length 2 list to {@code inputs}. The other arguments behave as in the analogous method on
     * ONNXContext. Output names are generated based on the {@link ONNXOperator#getOpName} and the name of the input nodes.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param other A second input argument to {@code op}
     * @param attributes A map of attributes of the operation, passed to {@link ONNXOperator#build(ONNXContext, String, String, Map)}.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op, ONNXRef<?> other, Map<String, Object> attributes) {
        return context.operation(op, Arrays.asList(this, other), getBaseName() + "_" + op.getOpName() + "_" + other.getBaseName(), attributes);
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String)}, passing this ONNXRef
     * and {@code other} as a length 2 list to {@code inputs}.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param other A second input argument to {@code op}
     * @param outputName A name that the output node of {@code op} will take.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op, ONNXRef<?> other, String outputName) {
        return context.operation(op, Arrays.asList(this, other), outputName, Collections.emptyMap());
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String, Map)}, passing this ONNXRef
     * and {@code other} as a length 2 list to {@code inputs}. Output names are generated based on the
     * {@link ONNXOperator#getOpName} and the name of the input nodes.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param other A second input argument to {@code op}
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op, ONNXRef<?> other) {
        return context.operation(op, Arrays.asList(this, other), getBaseName() + "_" + op.getOpName() + "_" + other.getBaseName(), Collections.emptyMap());
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String, Map)}, using this ONNXRef
     * as the first argument to {@code inputs}, with {@code otherInputs} append as subsequent arguments. Output names
     * are generated based on the {@link ONNXOperator#getOpName} and the name of the input nodes.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param others List of ONNXRefs supplied as inputs to {@code op} after this ONNXRef.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op, List<ONNXRef<?>> others) {
        return apply(op, others, Collections.singletonList(getBaseName() + "_" + others.stream().map(ONNXRef::getBaseName).collect(Collectors.joining("_"))), Collections.emptyMap()).get(0);
    }

    /**
     * Convenience method that calls {@link ONNXContext#operation(ONNXOperator, List, String, Map)}, using this ONNXRef
     * as the argument to {@code inputs}, with {@code otherInputs} append as subsequent arguments.
     * @param op An ONNXOperator to add to the graph, taking {@code inputs} as input.
     * @param others List of ONNXRefs supplied as inputs to {@code op} after this ONNXRef.
     * @param outputName The name for the constructed node.
     * @return a list of {@link ONNXNode}s that are the output nodes of {@code op}.
     */
    public ONNXNode apply(ONNXOperator op, List<ONNXRef<?>> others, String outputName) {
        return apply(op, others, Collections.singletonList(outputName), Collections.emptyMap()).get(0);
    }

    /**
     * Convenience method that calls {@link ONNXContext#assignTo(ONNXRef, ONNXRef)}, using this ONNXRef as the argument
     * to {@code input}.
     * @param output The output node / left-hand side of the assignment
     * @param <Ret> the {@link ONNXRef} type of the output.
     * @return the output node that was assigned to.
     */
    public <Ret extends ONNXRef<?>> Ret assignTo(Ret output) {
        return context.assignTo(this, output);
    }

    /**
     * Casts this ONNXRef to a different type using the {@link ONNXOperators#CAST} operation, and returning the output
     * node of that op. Currently supports only float, double, int, and long, which are specified by their respective
     * {@link Class} objects (e.g., {@code float.class}). Throws {@link IllegalArgumentException} when an unsupported cast
     * is requested.
     * @param clazz The class object specifying the type to cast to.
     * @return An ONNXRef representing this object cast into the requested type.
     */
    public ONNXNode cast(Class<?> clazz) {
        if (clazz.equals(float.class)) {
            return apply(ONNXOperators.CAST, Collections.singletonMap("to", OnnxMl.TensorProto.DataType.FLOAT.getNumber()));
        } else if(clazz.equals(double.class)) {
            return apply(ONNXOperators.CAST, Collections.singletonMap("to", OnnxMl.TensorProto.DataType.DOUBLE.getNumber()));
        } else if(clazz.equals(int.class)) {
            return apply(ONNXOperators.CAST, Collections.singletonMap("to", OnnxMl.TensorProto.DataType.INT32.getNumber()));
        } else if(clazz.equals(long.class)) {
            return apply(ONNXOperators.CAST, Collections.singletonMap("to", OnnxMl.TensorProto.DataType.INT64.getNumber()));
        } else {
            throw new IllegalArgumentException("unsupported class for casting: " + clazz.getName());
        }
    }
}

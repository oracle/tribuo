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

package org.tribuo.interop.tensorflow;

import com.google.protobuf.ByteString;
import org.tensorflow.Graph;
import org.tensorflow.GraphOperation;
import org.tensorflow.GraphOperationBuilder;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Scope;
import org.tensorflow.types.family.TType;
import org.tribuo.interop.tensorflow.protos.TensorTupleProto;
import org.tribuo.util.Util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Helper functions for working with TensorFlow.
 */
public abstract class TensorFlowUtil {
    private static final Logger logger = Logger.getLogger(TensorFlowUtil.class.getName());

    /**
     * The name of the variable op.
     */
    public static final String VARIABLE_V2 = "VariableV2";
    /**
     * The name of the assignment op.
     */
    public static final String ASSIGN_OP = "Assign";
    /**
     * The name given to the assignment operation from the placeholders.
     */
    public static final String ASSIGN_PLACEHOLDER = "Assign_from_Placeholder";
    /**
     * The name of the placeholder op.
     */
    public static final String PLACEHOLDER = "Placeholder";
    /**
     * The name of the data type.
     */
    public static final String DTYPE = "dtype";

    private TensorFlowUtil() {}

    /**
     * Closes a collection of {@link Tensor}s.
     *
     * @param tensors The collection of tensors to close.
     */
    public static void closeTensorCollection(Collection<Tensor> tensors) {
        for (Tensor t : tensors) {
            t.close();
        }
    }

    /**
     * Annotates a graph with an extra placeholder and assign operation for each
     * VariableV2. This allows the graph to be deserialised using {@link TensorFlowUtil#restoreMarshalledVariables(Session, Map)}.
     * <p>
     * This operation can either be done each time the Graph is loaded before deserialise is called,
     * or once, and the updated graphDef persisted with the Map produced by serialise.
     * <p>
     * Requires a session to correctly get the output type of a VariableV2.
     *
     * @param graph The graph to annotate.
     * @param session The session to use.
     */
    public static void annotateGraph(Graph graph, Session session) {
        List<String> variableNames = new ArrayList<>();
        Map<String, GraphOperation> opMap = new HashMap<>();
        Iterator<GraphOperation> opItr = graph.operations();
        while (opItr.hasNext()) {
            GraphOperation op = opItr.next();
            if (op.type().equals(VARIABLE_V2)) {
                variableNames.add(op.name());
                opMap.put(op.name(), op);
            }
        }

        Session.Runner runner = session.runner();
        for (String s : variableNames) {
            runner.fetch(s);
        }

        List<Tensor> output = runner.run();

        if (output.size() != variableNames.size()) {
            closeTensorCollection(output);
            throw new IllegalStateException("Failed to annotate all requested variables. Requested " + variableNames.size() + ", found " + output.size());
        }

        Scope scope = graph.baseScope();

        for (int i = 0; i < output.size(); i++) {
            GraphOperationBuilder builder = graph.opBuilder(PLACEHOLDER, generatePlaceholderName(variableNames.get(i)),scope);
            builder.setAttr(DTYPE, output.get(i).dataType());
            GraphOperation o = builder.build();
            builder = graph.opBuilder(ASSIGN_OP, variableNames.get(i) + "/" + ASSIGN_PLACEHOLDER,scope);
            builder.addInput(opMap.get(variableNames.get(i)).output(0));
            builder.addInput(o.output(0));
            builder.build();
        }

        closeTensorCollection(output);
    }

    /**
     * Creates a name for a placeholder based on the supplied variable name.
     *
     * @param variableName The variable name to use as a base.
     * @return A name for the placeholder.
     */
    public static String generatePlaceholderName(String variableName) {
        return variableName + "-tribuo-" + PLACEHOLDER;
    }

    /**
     * Extracts a Map containing the name of each Tensorflow VariableV2 and the
     * associated parameter array. This map can then be serialised to disk.
     *
     * @param graph The graph to read operations from.
     * @param session The session to read from.
     * @return A map containing all variable names and parameter arrays.
     */
    public static Map<String, TensorTuple> extractMarshalledVariables(Graph graph, Session session) {
        List<String> variableNames = new ArrayList<>();
        Iterator<GraphOperation> opItr = graph.operations();
        while (opItr.hasNext()) {
            GraphOperation op = opItr.next();
            if (op.type().equals(VARIABLE_V2)) {
                variableNames.add(op.name());
            }
        }

        Session.Runner runner = session.runner();
        for (String s : variableNames) {
            runner.fetch(s);
        }
        List<Tensor> output = runner.run();

        if (output.size() != variableNames.size()) {
            closeTensorCollection(output);
            throw new IllegalStateException("Failed to serialise all requested variables. Requested " + variableNames.size() + ", found " + output.size());
        }

        Map<String, TensorTuple> tensorMap = new HashMap<>();
        for (int i = 0; i < variableNames.size(); i++) {
            String name = variableNames.get(i);
            Tensor tensor = output.get(i);
            tensorMap.put(name, TensorTuple.of((TType)tensor));
        }

        closeTensorCollection(output);

        return tensorMap;
    }

    /**
     * Writes a map containing the name of each Tensorflow VariableV2 and the associated
     * parameter array into the supplied session.
     *
     * @param session   The session to write to.
     * @param tensorMap The parameter map to write.
     */
    public static void restoreMarshalledVariables(Session session, Map<String, TensorTuple> tensorMap) {
        Session.Runner runner = session.runner();
        List<Tensor> tensors = new ArrayList<>();
        for (Map.Entry<String, TensorTuple> e : tensorMap.entrySet()) {
            logger.log(Level.FINEST, "Loading " + e.getKey() + " of type " + e.getValue().getClass().getName());
            Tensor tensor = e.getValue().rebuildTensor();
            runner.feed(generatePlaceholderName(e.getKey()), tensor);
            runner.addTarget(e.getKey() + "/" + ASSIGN_PLACEHOLDER);
            tensors.add(tensor);
        }
        runner.run();
        closeTensorCollection(tensors);
    }

    /**
     * A serializable tuple containing the tensor class name, the shape and the data.
     * <p>
     * It's almost a record.
     */
    public static final class TensorTuple implements Serializable {
        private static final long serialVersionUID = 1L;

        /**
         * The tensor class name.
         */
        public final String className;
        /**
         * The shape of the tensor.
         */
        public final long[] shape;
        /**
         * The tensor data.
         */
        public final byte[] data;

        /**
         * Makes a TensorTuple.
         * @param className The tensor class name.
         * @param shape The dimensions of the tensor.
         * @param data The data in the tensor.
         */
        public TensorTuple(String className, long[] shape, byte[] data) {
            this.className = className;
            this.shape = shape;
            this.data = data;
        }

        /**
         * Deserializes the tensor tuple from the supplied protobuf.
         * @param proto The proto to deserialize.
         */
        public TensorTuple(TensorTupleProto proto) {
            this.className = proto.getClassName();
            this.shape = Util.toPrimitiveLong(proto.getShapeList());
            this.data = proto.getData().toByteArray();
        }

        /**
         * Recreates the Tensor from the serialized form.
         * @return The Tensor.
         */
        public Tensor rebuildTensor() {
            try {
                Class<?> clazz = Class.forName(className);
                if (TType.class.isAssignableFrom(clazz)) {
                    @SuppressWarnings("unchecked") // guarded by if
                    Class<? extends TType> tensorClass = (Class<? extends TType>) clazz;
                    Shape shapeObj = Shape.of(shape);
                    ByteDataBuffer buf = DataBuffers.of(data);
                    return Tensor.of(tensorClass,shapeObj,buf);
                } else {
                    throw new IllegalStateException("Unexpected Tensor type, found " + className);
                }
            } catch (ClassNotFoundException e) {
                throw new IllegalStateException("Failed to instantiate Tensor class",e);
            }
        }

        /**
         * Serializes this object to a protobuf.
         * @return The protobuf.
         */
        public TensorTupleProto serialize() {
            TensorTupleProto.Builder builder = TensorTupleProto.newBuilder();

            builder.setClassName(className);
            builder.addAllShape(Arrays.stream(shape).boxed().collect(Collectors.toList()));
            builder.setData(ByteString.copyFrom(data));

            return builder.build();
        }

        /**
         * Makes a TensorTuple out of this tensor.
         * @param tensor The tensor to serialize.
         * @return A serializable form of the Tensor.
         */
        public static TensorTuple of(TType tensor) {
            ByteDataBuffer buffer = tensor.asRawTensor().data();
            long size = buffer.size();
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Cannot serialize Tensors bigger than Integer.MAX_VALUE, found " + size);
            }
            String className = tensor.type().getName();
            long[] shape = tensor.shape().asArray();
            byte[] data = new byte[(int)size];
            buffer.read(data);

            return new TensorTuple(className,shape,data);
        }
    }
}

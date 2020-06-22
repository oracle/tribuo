/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Helper functions for working with Tensorflow.
 */
public class TensorflowUtil {
    private static final Logger logger = Logger.getLogger(TensorflowUtil.class.getName());

    public static final String VARIABLE_V2 = "VariableV2";
    public static final String ASSIGN_OP = "Assign";
    public static final String ASSIGN_PLACEHOLDER = "Assign_from_Placeholder";
    public static final String PLACEHOLDER = "Placeholder";
    public static final String DTYPE = "dtype";

    /**
     * Creates a new primitive boolean array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A boolean array.
     */
    public static Object newBooleanArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new boolean[(int) shape[0]];
            case 2:
                return new boolean[(int) shape[0]][(int) shape[1]];
            case 3:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive byte array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A byte array.
     */
    public static Object newByteArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new byte[(int) shape[0]];
            case 2:
                return new byte[(int) shape[0]][(int) shape[1]];
            case 3:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive int array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A int array.
     */
    public static Object newIntArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new int[(int) shape[0]];
            case 2:
                return new int[(int) shape[0]][(int) shape[1]];
            case 3:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive long array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A long array.
     */
    public static Object newLongArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new long[(int) shape[0]];
            case 2:
                return new long[(int) shape[0]][(int) shape[1]];
            case 3:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive float array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A float array.
     */
    public static Object newFloatArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new float[(int) shape[0]];
            case 2:
                return new float[(int) shape[0]][(int) shape[1]];
            case 3:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive double array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A double array.
     */
    public static Object newDoubleArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new double[(int) shape[0]];
            case 2:
                return new double[(int) shape[0]][(int) shape[1]];
            case 3:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Closes a list of {@link Tensor}s.
     *
     * @param tensorList The list of tensors to close.
     */
    public static void closeTensorList(List<Tensor<?>> tensorList) {
        for (Tensor<?> t : tensorList) {
            t.close();
        }
    }

    /**
     * Extracts the appropriate type of primitive array from a {@link Tensor}.
     * <p>
     * Returns an object as the user doesn't know what type is in the {@link Tensor}.
     *
     * @param tensor The tensor to read.
     * @return A primitive array.
     */
    public static Object convertTensorToArray(Tensor<?> tensor) {
        long[] shape = tensor.shape();

        Object array;
        switch (tensor.dataType()) {
            case FLOAT:
                array = newFloatArray(shape);
                break;
            case DOUBLE:
                array = newDoubleArray(shape);
                break;
            case INT32:
                array = newIntArray(shape);
                break;
            case UINT8:
            case STRING:
                array = newByteArray(shape);
                break;
            case INT64:
                array = newLongArray(shape);
                break;
            case BOOL:
                array = newBooleanArray(shape);
                break;
            default:
                throw new IllegalArgumentException("Tribuo can't serialise Tensors with type " + tensor.dataType());
        }

        tensor.copyTo(array);

        return array;
    }

    /**
     * Converts a {@link Tensor} into a scalar object, boxing the primitive types.
     * <p>
     * Does not close the Tensor.
     *
     * @param tensor The tensor to convert.
     * @return A boxed scalar.
     */
    public static Object convertTensorToScalar(Tensor<?> tensor) {
        Object scalar;
        switch (tensor.dataType()) {
            case FLOAT:
                scalar = tensor.floatValue();
                break;
            case DOUBLE:
                scalar = tensor.doubleValue();
                break;
            case INT32:
                scalar = tensor.intValue();
                break;
            case UINT8:
                scalar = (byte) (tensor.intValue() & 0xFF);
                break;
            case STRING:
                scalar = tensor.bytesValue();
                break;
            case INT64:
                scalar = tensor.longValue();
                break;
            case BOOL:
                scalar = tensor.booleanValue();
                break;
            default:
                throw new IllegalArgumentException("Tribuo can't serialise Tensors with type " + tensor.dataType());
        }
        return scalar;
    }

    /**
     * Annotates a graph with an extra placeholder and assign operation for each
     * VariableV2. This allows the graph to be deserialised using {@link TensorflowUtil#deserialise(Session, Map)}.
     * <p>
     * This operation can either be done each time the Graph is loaded before deserialise is called,
     * or once, and the updated graphDef persisted with the Map produced by serialise.
     * <p>
     * Requires a session to correctly get the output type of a VariableV2. This isn't strictly necessary,
     * but there aren't typed ways to get outputs in the TF version we use.
     *
     * @param graph The graph to annotate.
     * @param session The session to use.
     */
    public static void annotateGraph(Graph graph, Session session) {
        List<String> variableNames = new ArrayList<>();
        Map<String, Operation> opMap = new HashMap<>();
        Iterator<Operation> opItr = graph.operations();
        while (opItr.hasNext()) {
            Operation op = opItr.next();
            if (op.type().equals(VARIABLE_V2)) {
                variableNames.add(op.name());
                opMap.put(op.name(), op);
            }
        }

        Session.Runner runner = session.runner();
        for (String s : variableNames) {
            runner.fetch(s);
        }

        List<Tensor<?>> output = runner.run();

        if (output.size() != variableNames.size()) {
            TensorflowUtil.closeTensorList(output);
            throw new IllegalStateException("Failed to annotate all requested variables. Requested " + variableNames.size() + ", found " + output.size());
        }

        for (int i = 0; i < output.size(); i++) {
            OperationBuilder builder = graph.opBuilder(PLACEHOLDER, generatePlaceholderName(variableNames.get(i)));
            builder.setAttr(DTYPE, output.get(i).dataType());
            Operation o = builder.build();
            builder = graph.opBuilder(ASSIGN_OP, variableNames.get(i) + "/" + ASSIGN_PLACEHOLDER);
            builder.addInput(opMap.get(variableNames.get(i)).output(0));
            builder.addInput(o.output(0));
            builder.build();
        }

        TensorflowUtil.closeTensorList(output);
    }

    /**
     * Creates a name for a placeholder based on the supplied variable name.
     *
     * @param variableName The variable name to use as a base.
     * @return A name for the placeholder.
     */
    public static String generatePlaceholderName(String variableName) {
        return variableName + "-" + PLACEHOLDER;
    }

    /**
     * Extracts a Map containing the name of each Tensorflow VariableV2 and the
     * associated parameter array. This map can then be serialised to disk.
     *
     * @param graph The graph to read operations from.
     * @param session The session to read from.
     * @return A map containing all variable names and parameter arrays.
     */
    public static Map<String, Object> serialise(Graph graph, Session session) {
        List<String> variableNames = new ArrayList<>();
        Iterator<Operation> opItr = graph.operations();
        while (opItr.hasNext()) {
            Operation op = opItr.next();
            if (op.type().equals(VARIABLE_V2)) {
                variableNames.add(op.name());
            }
        }

        Session.Runner runner = session.runner();
        for (String s : variableNames) {
            runner.fetch(s);
        }
        List<Tensor<?>> output = runner.run();

        if (output.size() != variableNames.size()) {
            closeTensorList(output);
            throw new IllegalStateException("Failed to serialise all requested variables. Requested " + variableNames.size() + ", found " + output.size());
        }

        Map<String, Object> tensorMap = new HashMap<>();
        for (int i = 0; i < variableNames.size(); i++) {
            String name = variableNames.get(i);
            Tensor<?> tensor = output.get(i);
            Object value;
            if (tensor.numDimensions() == 0) {
                value = convertTensorToScalar(tensor);
            } else {
                value = convertTensorToArray(tensor);
            }
            tensorMap.put(name, value);
        }

        closeTensorList(output);

        return tensorMap;
    }

    /**
     * Writes a map containing the name of each Tensorflow VariableV2 and the associated
     * parameter array into the supplied session.
     *
     * @param session   The session to write to.
     * @param tensorMap The parameter map to write.
     */
    public static void deserialise(Session session, Map<String, Object> tensorMap) {
        Session.Runner runner = session.runner();
        List<Tensor<?>> tensors = new ArrayList<>();
        for (Map.Entry<String, Object> e : tensorMap.entrySet()) {
            logger.log(Level.FINEST, "Loading " + e.getKey() + " of type " + e.getValue().getClass().getName());
            Tensor<?> tensor = Tensor.create(e.getValue());
            runner.feed(generatePlaceholderName(e.getKey()), tensor);
            runner.addTarget(e.getKey() + "/" + ASSIGN_PLACEHOLDER);
            tensors.add(tensor);
        }
        runner.run();
        closeTensorList(tensors);
    }
}

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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.tribuo.util.onnx.ONNXAttribute.VARIADIC_INPUT;

/**
 * ONNX Opset 13, and ONNX-ML version 1.
 * <p>
 * In a future version of Tribuo this class will be split into two enums, one for ONNX opset 13 and one for ONNX-ML v1.
 */
public enum ONNXOperators implements ONNXOperator {
    /**
     * Identity.
     */
    IDENTITY("Identity",1,1),
    /**
     * Concatenates tensors.
     */
    CONCAT("Concat",VARIADIC_INPUT,1, Collections.singletonList(
            new ONNXAttribute("axis", OnnxMl.AttributeProto.AttributeType.INT, true)
    )),
    /**
     * Makes a constant of the size of the supplied shape containing the {@code value}.
     */
    CONSTANT_OF_SHAPE("ConstantOfShape",1,1, Collections.singletonList(
            new ONNXAttribute("value", OnnxMl.AttributeProto.AttributeType.TENSOR, false)
    )),
    /**
     * Sigmoid element-wise.
     */
    SIGMOID("Sigmoid",1,1),
    /**
     * Softmax.
     * <ul>
     *     <li>{@code axis} defaults to -1.</li>
     * </ul>
     */
    SOFTMAX("Softmax",1,1, Collections.singletonList(
            new ONNXAttribute("axis", OnnxMl.AttributeProto.AttributeType.INT, false)
    )),
    /**
     * Cast input to specified type.
     * <ul>
     *     <li>{@code to} must be a data type int from {@link OnnxMl.TensorProto.DataType}</li>
     * </ul>
     */
    CAST("Cast",1,1, Collections.singletonList(
            new ONNXAttribute("to", OnnxMl.AttributeProto.AttributeType.INT, true)
    )),
    /**
     * Element-wise negation.
     */
    NEG("Neg",1,1),
    /**
     * Element-wise addition with broadcasting.
     */
    ADD("Add",2,1),
    /**
     * Element-wise subtraction with broadcasting.
     */
    SUB("Sub",2,1),
    /**
     * Element-wise multiplication with broadcasting.
     */
    MUL("Mul",2,1),
    /**
     * Element-wise division with broadcasting.
     */
    DIV("Div",2,1),
    /**
     * Element-wise exponentiation with broadcasting.
     */
    POW("Pow",2,1),
    /**
     * Element-wise summation across the supplied inputs with broadcasting.
     */
    SUM("Sum",VARIADIC_INPUT,1),
    /**
     * Gathers elements from the first argument (of rank r) indexed by the second argument (of rank q) producing
     * a tensor of rank {@code q + r - 1}.
     * <ul>
     *     <li>{@code axis} the axis of the first argument to gather from.</li>
     * </ul>
     */
    GATHER("Gather",2,1,Collections.singletonList(
            new ONNXAttribute("axis", OnnxMl.AttributeProto.AttributeType.INT, true)
    )),
    /**
     * Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise.
     * <ul>
     *     <li>{@code axis} default is -1, i.e., take the hardmax over the last dimension.</li>
     * </ul>
     */
    HARDMAX("Hardmax",1,1, Collections.singletonList(
            new ONNXAttribute("axis", OnnxMl.AttributeProto.AttributeType.INT,false)
    )),
    /**
     * Computes the mean of the input tensor's element along the provided axes.
     * <ul>
     *     <li>{@code axes} default is to reduce over all dimensions.</li>
     *     <li>{@code keepdims} defaults to 1 which means keep.</li>
     * </ul>
     */
    REDUCE_MEAN("ReduceMean",1,1,Arrays.asList(
            new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS,false),
            new ONNXAttribute("keepdims", OnnxMl.AttributeProto.AttributeType.INT,false)
    )),
    /**
     * Compute the minimum along the specified axes of the tensor.
     * <ul>
     *     <li>{@code axes} defaults to all dimensions.</li>
     *     <li>{@code keepdims} defaults to 1 which means keep.</li>
     * </ul>
     */
    REDUCE_MIN("ReduceMin",1,1,Arrays.asList(
            new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS,false),
            new ONNXAttribute("keepdims", OnnxMl.AttributeProto.AttributeType.INT,false)
    )),
    /**
     * Compute the sum along the specified axes of the tensor, the axes are the second input.
     * <ul>
     *     <li>{@code keepdims} defaults to 1 which means keep.</li>
     *     <li>{@code noop_with_empty_axes} defaults to 0 which means empty axes reduces the tensor to a scalar.</li>
     * </ul>
     */
    REDUCE_SUM("ReduceSum",1,1,1,Arrays.asList(
            // Opset 11 version: new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS, false),
            new ONNXAttribute("keepdims", OnnxMl.AttributeProto.AttributeType.INT, false),
            new ONNXAttribute("noop_with_empty_axes", OnnxMl.AttributeProto.AttributeType.INT, false) // Opset 13
    )),
    /**
     * Adds extra dimensions to a tensor in the specified places, the axes are the second input.
     */
    UNSQUEEZE("Unsqueeze",2,1),
    // Unsqueeze Opset 11: Collections.singletonList(new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS, true))
    /**
     * General Matrix Multiply: {@code alpha*AB + beta*C}.
     * <p>
     * The {@code C} input is optional, and if not supplied is treated as zero.
     * <ul>
     *     <li>{@code alpha} defaults to 1.0</li>
     *     <li>{@code beta} defaults to 1.0</li>
     *     <li>{@code transA} defaults to 0 (i.e., not transposed)</li>
     *     <li>{@code transB} defaults to 0 (i.e., not transposed)</li>
     * </ul>
     */
    GEMM("Gemm",2,1, 1, Arrays.asList(
            new ONNXAttribute("alpha", OnnxMl.AttributeProto.AttributeType.FLOAT,false),
            new ONNXAttribute("beta", OnnxMl.AttributeProto.AttributeType.FLOAT,false),
            new ONNXAttribute("transA", OnnxMl.AttributeProto.AttributeType.INT,false),
            new ONNXAttribute("transB", OnnxMl.AttributeProto.AttributeType.INT,false)
    )),
    /**
     * Greater than, returns the element-wise greater than operation on the two tensors.
     * <p>
     * Tensors must be broadcastable to the same shape.
     */
    GREATER("Greater",2,1),
    /**
     * Greater than or equal to, returns the element-wise greater than or equal to operation on the two tensors.
     * <p>
     * Tensors must be broadcastable to the same shape.
     */
    GREATER_OR_EQUAL("GreaterOrEqual",2,1),
    /**
     * Less than, returns the element-wise less than operation on the two tensors.
     * <p>
     * Tensors must be broadcastable to the same shape.
     */
    LESS("Less",2,1),
    /**
     * Less than or equal to, returns the element-wise less than or equal to operation on the two tensors.
     * <p>
     * Tensors must be broadcastable to the same shape.
     */
    LESS_OR_EQUAL("LessOrEqual",2,1),
    /**
     * Choice operator, based on the true value of the condition input, returns the element at that index from either
     * the second or third input. When the test is true, return the second input, otherwise return the third input.
     */
    WHERE("Where",3,1),
    /**
     * Array feature extractor, selects the indices specified by the second tensor from the last dimension of the first tensor.
     */
    ARRAY_FEATURE_EXTRACTOR("ArrayFeatureExtractor",2,1,"ai.onnx.ml"),
    /**
     * SVM Classifier.
     * <ul>
     *     <li>{@code classlabels_ints} - Class labels if using integer labels. One and only one of the 'classlabels_*' attributes must be defined.</li>
     *     <li>{@code classlabels_strings} - Class labels if using string labels. One and only one of the 'classlabels_*' attributes must be defined.</li>
     *     <li>{@code coefficients} - SVM coefficients</li>
     *     <li>{@code kernel_params} - Tuple of gamma, coef0 and degree. Set to zero if unused by the kernel.</li>
     *     <li>{@code kernel_type} - One of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.</li>
     *     <li>{@code post_transforms} - Transform to apply to the score (usually unused by SVMs), one of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'.</li>
     *     <li>{@code prob_a} - Probability coefficients, if set must be the same size as prob_b.</li>
     *     <li>{@code prob_b} - Probability coefficients, if set must be the same size as prob_a.</li>
     *     <li>{@code rho} - Rho vector.</li>
     *     <li>{@code support_vectors} - linearised support vectors.</li>
     *     <li>{@code vectors_per_class} - the number of support vectors in each class.</li>
     * </ul>
     */
    SVM_CLASSIFIER("SVMClassifier",1,2, "ai.onnx.ml", Arrays.asList(
            new ONNXAttribute("classlabels_ints",OnnxMl.AttributeProto.AttributeType.INTS,false),
            new ONNXAttribute("classlabels_strings",OnnxMl.AttributeProto.AttributeType.STRINGS,false),
            new ONNXAttribute("coefficients",OnnxMl.AttributeProto.AttributeType.FLOATS,true),
            new ONNXAttribute("kernel_params",OnnxMl.AttributeProto.AttributeType.FLOATS,true),
            new ONNXAttribute("kernel_type",OnnxMl.AttributeProto.AttributeType.STRING,false),
            new ONNXAttribute("post_transform",OnnxMl.AttributeProto.AttributeType.STRING,false),
            new ONNXAttribute("prob_a",OnnxMl.AttributeProto.AttributeType.FLOATS,false),
            new ONNXAttribute("prob_b",OnnxMl.AttributeProto.AttributeType.FLOATS,false),
            new ONNXAttribute("rho",OnnxMl.AttributeProto.AttributeType.FLOATS,true),
            new ONNXAttribute("support_vectors",OnnxMl.AttributeProto.AttributeType.FLOATS,true),
            new ONNXAttribute("vectors_per_class",OnnxMl.AttributeProto.AttributeType.INTS,true)
    )),
    /**
     * SVM Regressor.
     * <ul>
     *     <li>{@code coefficients} - SVM coefficients</li>
     *     <li>{@code kernel_params} - Tuple of gamma, coef0 and degree. Set to zero if unused by the kernel.</li>
     *     <li>{@code kernel_type} - One of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.</li>
     *     <li>{@code n_supports} - The number of support vectors.</li>
     *     <li>{@code one_class} - Flag noting if this regression is a one-class SVM for anomaly detection or not.</li>
     *     <li>{@code post_transforms} - Transform to apply to the score (usually unused by SVMs), one of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'.</li>
     *     <li>{@code rho} - Rho vector.</li>
     *     <li>{@code support_vectors} - linearised support vectors.</li>
     * </ul>
     */
    SVM_REGRESSOR("SVMRegressor",1,1, "ai.onnx.ml", Arrays.asList(
            new ONNXAttribute("coefficients",OnnxMl.AttributeProto.AttributeType.FLOATS,true),
            new ONNXAttribute("kernel_params",OnnxMl.AttributeProto.AttributeType.FLOATS,true),
            new ONNXAttribute("kernel_type",OnnxMl.AttributeProto.AttributeType.STRING,false),
            new ONNXAttribute("n_supports",OnnxMl.AttributeProto.AttributeType.INT,true),
            new ONNXAttribute("one_class",OnnxMl.AttributeProto.AttributeType.INT,false),
            new ONNXAttribute("post_transform",OnnxMl.AttributeProto.AttributeType.STRING,false),
            new ONNXAttribute("rho",OnnxMl.AttributeProto.AttributeType.FLOATS,true),
            new ONNXAttribute("support_vectors",OnnxMl.AttributeProto.AttributeType.FLOATS,true)
    ))
    ;

    /**
     * The operator name.
     */
    public final String opName;
    /**
     * The number of inputs.
     */
    public final int numInputs;
    /**
     * The number of optional inputs.
     */
    public final int numOptionalInputs;
    /**
     * The number of outputs.
     */
    public final int numOutputs;
    /**
     * The operator attributes.
     */
    public final Map<String,ONNXAttribute> attributes;
    /**
     * The mandatory attribute names.
     */
    public final Set<String> mandatoryAttributeNames;
    /**
     * The operator domain (used for the ML operators).
     * <p>
     * Null if the domain is the default one.
     */
    public final String domain;

    /**
     * Opset supported by these definitions.
     */
    private static final int OPSET_VERSION = 13;

    /**
     * Builds an operator without attributes.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOutputs The number of outputs.
     */
    private ONNXOperators(String value, int numInputs, int numOutputs) {
        this(value,numInputs,0,numOutputs);
    }

    /**
     * Builds an operator without attributes and with optional inputs.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOptionalInputs The number of optional inputs.
     * @param numOutputs The number of outputs.
     */
    private ONNXOperators(String value, int numInputs, int numOptionalInputs, int numOutputs) {
        this(value, numInputs, numOptionalInputs, numOutputs, (String) null);
    }

    /**
     * Builds an operator without attributes and with optional inputs.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOutputs The number of outputs.
     * @param domain The domain.
     */
    private ONNXOperators(String value, int numInputs, int numOutputs, String domain) {
        this(value, numInputs, 0, numOutputs, domain);
    }

    /**
     * Builds an operator without attributes and with optional inputs.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOptionalInputs The number of optional inputs.
     * @param numOutputs The number of outputs.
     * @param domain The domain.
     */
    private ONNXOperators(String value, int numInputs, int numOptionalInputs, int numOutputs, String domain) {
        this.opName = value;
        this.numInputs = numInputs;
        this.numOptionalInputs = numOptionalInputs;
        this.numOutputs = numOutputs;
        this.attributes = Collections.emptyMap();
        this.mandatoryAttributeNames = Collections.emptySet();
        this.domain = domain;
    }

    /**
     * Builds an operator with attributes.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOutputs The number of outputs.
     * @param attributes The attributes.
     */
    private ONNXOperators(String value, int numInputs, int numOutputs, List<ONNXAttribute> attributes) {
        this(value,numInputs,0,numOutputs,attributes);
    }

    /**
     * Builds an operator with attributes.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOutputs The number of outputs.
     * @param domain The domain.
     * @param attributes The attributes.
     */
    private ONNXOperators(String value, int numInputs, int numOutputs, String domain, List<ONNXAttribute> attributes) {
        this(value,numInputs,0,numOutputs, domain, attributes);
    }

    /**
     * Builds an operator with attributes and optional inputs.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOptionalInputs The number of optional inputs.
     * @param numOutputs The number of outputs.
     * @param attributes The attributes.
     */
    private ONNXOperators(String value, int numInputs, int numOptionalInputs, int numOutputs, List<ONNXAttribute> attributes) {
        this(value,numInputs,numOptionalInputs,numOutputs, null, attributes);
    }

    /**
     * Builds an operator with attributes and optional inputs.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOptionalInputs The number of optional inputs.
     * @param numOutputs The number of outputs.
     * @param domain The operator domain.
     * @param attributes The attributes.
     */
    private ONNXOperators(String value, int numInputs, int numOptionalInputs, int numOutputs, String domain, List<ONNXAttribute> attributes) {
        this.opName = value;
        this.numInputs = numInputs;
        this.numOptionalInputs = numOptionalInputs;
        this.numOutputs = numOutputs;
        Map<String,ONNXAttribute> attributeMap = new HashMap<>();
        Set<String> attributeSet = new HashSet<>();
        for (ONNXAttribute a : attributes) {
            attributeMap.put(a.getName(),a);
            if (a.isMandatory()) {
                attributeSet.add(a.getName());
            }
        }
        if (attributes.size() != attributeMap.size()) {
            throw new IllegalArgumentException("Duplicate attribute in enum declaration - " + attributes);
        }
        this.attributes = Collections.unmodifiableMap(attributeMap);
        this.mandatoryAttributeNames = attributeSet.isEmpty() ? Collections.emptySet() : Collections.unmodifiableSet(attributeSet);
        this.domain = domain;
    }

    @Override
    public String getOpName() {
        return opName;
    }

    @Override
    public int getNumInputs() {
        return numInputs;
    }

    @Override
    public int getNumOptionalInputs() {
        return numOptionalInputs;
    }

    @Override
    public int getNumOutputs() {
        return numOutputs;
    }

    @Override
    public Map<String,ONNXAttribute> getAttributes() {
        return attributes;
    }

    @Override
    public Set<String> getMandatoryAttributeNames() {
        return mandatoryAttributeNames;
    }

    @Override
    public int getOpVersion() {
        return getOpsetVersion();
    }

    @Override
    public String getOpDomain() {
        return domain;
    }

    /**
     * Returns the opset version supported by these operators.
     * @return The opset version.
     */
    public static int getOpsetVersion() {
        return OPSET_VERSION;
    }

    /**
     * Returns the opset proto for these operators.
     * @return The opset proto.
     */
    public static OnnxMl.OperatorSetIdProto getOpsetProto() {
        return OnnxMl.OperatorSetIdProto.newBuilder().setVersion(ONNXOperators.getOpsetVersion()).build();
    }
}

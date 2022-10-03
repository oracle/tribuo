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

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.framework.op.FrameworkOps;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.TFloat16;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.interop.tensorflow.protos.OutputConverterProto;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.logging.Logger;

/**
 * Can convert a {@link MultiLabel} into a {@link Tensor} containing a binary encoding of the label vector and
 * can convert a {@link TFloat16} or {@link TFloat32} into a {@link Prediction} or a {@link MultiLabel}.
 * <p>
 * Predictions are thresholded at {@link #THRESHOLD}, probabilities above this are considered to be present in the
 * output.
 */
@ProtoSerializableClass(version = MultiLabelConverter.CURRENT_VERSION)
public class MultiLabelConverter implements OutputConverter<MultiLabel> {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(MultiLabelConverter.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * The threshold to determine if a label has been predicted.
     */
    public static final double THRESHOLD = 0.5;

    /**
     * Constructs a MultiLabelConverter.
     */
    public MultiLabelConverter() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static MultiLabelConverter deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new MultiLabelConverter();
    }

    @Override
    public OutputConverterProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Returns a sigmoid cross-entropy loss.
     * @return The sigmoid cross-entropy loss.
     */
    @Override
    public BiFunction<Ops, Pair<Placeholder<? extends TNumber>,Operand<TNumber>>,Operand<TNumber>> loss() {
        return (ops,pair) -> {
            FrameworkOps frameworkOps = FrameworkOps.create(ops);
            @SuppressWarnings("unchecked") // cast off the wildcard to the superclass
            Placeholder<TNumber> placeholder = (Placeholder<TNumber>) pair.getA();
            return ops.math.mean(frameworkOps.nn.sigmoidCrossEntropyWithLogits(placeholder,pair.getB()),ops.constant(0));
        };
    }

    /**
     * Applies a softmax.
     * @param <V> The softmax input type (should be TFloat32).
     * @return A function which applies a softmax.
     */
    @Override
    public <V extends TNumber> BiFunction<Ops, Operand<V>, Op> outputTransformFunction() {
        return (ops, logits) -> ops.math.sigmoid(logits);
    }

    @Override
    public Prediction<MultiLabel> convertToPrediction(Tensor tensor, ImmutableOutputInfo<MultiLabel> outputIDInfo, int numValidFeatures, Example<MultiLabel> example) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        long batchSize = predictions.shape().asArray()[0];
        if (batchSize != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, batchSize = " + batchSize);
        }
        return generatePrediction(predictions.slice(Indices.at(0),Indices.all()),outputIDInfo,numValidFeatures,example);
    }

    private Prediction<MultiLabel> generatePrediction(FloatNdArray predictions, ImmutableOutputInfo<MultiLabel> outputIDInfo, int numUsed, Example<MultiLabel> example) {
        long[] shape = predictions.shape().asArray();
        if (shape.length != 1) {
            throw new IllegalArgumentException("Failed to get scalar predictions. Found " + Arrays.toString(shape));
        }
        if (shape[0] > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("More than Integer.MAX_VALUE predictions. Found " + shape[0]);
        }
        int length = (int) shape[0];
        Map<String,MultiLabel> fullLabels = new HashMap<>(outputIDInfo.size());
        Set<Label> predictedLabels = new HashSet<>();
        for (int i = 0; i < length; i++) {
            String labelName = outputIDInfo.getOutput(i).getLabelString();
            double labelScore = predictions.getFloat(i);
            Label score = new Label(labelName,labelScore);
            if (labelScore > THRESHOLD) {
                predictedLabels.add(score);
            }
            fullLabels.put(labelName,new MultiLabel(score));
        }
        return new Prediction<>(new MultiLabel(predictedLabels), fullLabels, numUsed, example, true);
    }

    @Override
    public MultiLabel convertToOutput(Tensor tensor, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        long batchSize = predictions.shape().asArray()[0];
        if (batchSize != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, batchSize = " + batchSize);
        }
        return generateMultiLabel(predictions.slice(Indices.at(0),Indices.all()),outputIDInfo);
    }

    private MultiLabel generateMultiLabel(FloatNdArray predictions, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        long[] shape = predictions.shape().asArray();
        if (shape.length != 1) {
            throw new IllegalArgumentException("Failed to get scalar predictions. Found " + Arrays.toString(shape));
        }
        if (shape[0] > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("More than Integer.MAX_VALUE predictions. Found " + shape[0]);
        }
        int length = (int) shape[0];
        Set<Label> predictedLabels = new HashSet<>();
        for (int i = 0; i < length; i++) {
            double labelScore = predictions.getFloat(i);
            Label score = new Label(outputIDInfo.getOutput(i).getLabelString(),labelScore);
            if (labelScore > THRESHOLD) {
                predictedLabels.add(score);
            }
        }
        return new MultiLabel(predictedLabels);
    }

    private FloatNdArray getBatchPredictions(Tensor tensor, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        long[] shape = tensor.shape().asArray();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Supplied tensor has the wrong number of dimensions, shape = " + Arrays.toString(shape));
        }
        int numValues = (int) shape[1];
        if (numValues != outputIDInfo.size()) {
            throw new IllegalArgumentException("Supplied tensor has incorrect number of elements, tensor output dimension: " + numValues + ", outputInfo dimension: " + outputIDInfo.size());
        }
        if (tensor instanceof TFloat16) {
            return (TFloat16) tensor;
        } else if (tensor instanceof TFloat32) {
            return (TFloat32) tensor;
        } else {
            throw new IllegalArgumentException("Tensor is not a probability distribution. Found type " + tensor.getClass().getName());
        }
    }

    @Override
    public List<Prediction<MultiLabel>> convertToBatchPrediction(Tensor tensor, ImmutableOutputInfo<MultiLabel> outputIDInfo, int[] numValidFeatures, List<Example<MultiLabel>> examples) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        List<Prediction<MultiLabel>> output = new ArrayList<>();

        int batchSize = (int) predictions.shape().asArray()[0];

        if ((batchSize != examples.size()) || (batchSize != numValidFeatures.length)) {
            throw new IllegalArgumentException("Invalid number of predictions received from Tensorflow, expected " + numValidFeatures.length + ", received " + batchSize);
        }

        for (int i = 0; i < batchSize; i++) {
            FloatNdArray slice = predictions.slice(Indices.at(i),Indices.all());
            output.add(generatePrediction(slice,outputIDInfo,numValidFeatures[i],examples.get(i)));
        }

        return output;
    }

    @Override
    public List<MultiLabel> convertToBatchOutput(Tensor tensor, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        List<MultiLabel> output = new ArrayList<>();

        int batchSize = (int) predictions.shape().asArray()[0];

        for (int i = 0; i < batchSize; i++) {
            FloatNdArray slice = predictions.slice(Indices.at(i),Indices.all());
            output.add(generateMultiLabel(slice,outputIDInfo));
        }

        return output;
    }

    @Override
    public Tensor convertToTensor(MultiLabel example, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        SparseVector vec = example.convertToSparseVector(outputIDInfo);
        TFloat32 returnVal = TFloat32.tensorOf(Shape.of(1,outputIDInfo.size()));
        for (int j = 0; j < outputIDInfo.size(); j++) {
            returnVal.setFloat(0.0f,0,j);
        }
        for (VectorTuple v : vec) {
            returnVal.setFloat((float)v.value, 0, v.index);
        }
        return returnVal;
    }

    @Override
    public Tensor convertToTensor(List<Example<MultiLabel>> examples, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        TFloat32 returnVal = TFloat32.tensorOf(Shape.of(examples.size(),outputIDInfo.size()));
        int i = 0;
        for (Example<MultiLabel> e : examples) {
            SparseVector vec = e.getOutput().convertToSparseVector(outputIDInfo);
            for (int j = 0; j < outputIDInfo.size(); j++) {
                returnVal.setFloat(0.0f,i,j);
            }
            for (VectorTuple v : vec) {
                returnVal.setFloat((float)v.value, i, v.index);
            }
            i++;
        }
        return returnVal;
    }

    @Override
    public boolean generatesProbabilities() {
        return true;
    }

    @Override
    public String toString() {
        return "MultiLabelConverter()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OutputConverter");
    }

    @Override
    public Class<MultiLabel> getTypeWitness() {
        return MultiLabel.class;
    }
}

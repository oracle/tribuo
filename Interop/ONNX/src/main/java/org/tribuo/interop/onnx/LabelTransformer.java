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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.SequenceInfo;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.interop.onnx.protos.LabelTransformerProto;
import org.tribuo.interop.onnx.protos.OutputTransformerProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Can convert an {@link OnnxValue} into a {@link Prediction} or a {@link Label}.
 * <p>
 * Accepts:
 * <ul>
 *     <li>a tuple (tensor,sequence(map(long,float))) - as produced by sk2onnx.</li>
 *     <li>a tuple (tensor, float tensor) - as produced by the bare ONNX ML operations (e.g., SVMClassifier).</li>
 *     <li>a single float tensor - as produced by pytorch.</li>
 * </ul>
 * By default it assumes the model scores are probabilities.
 * <p>
 * The scores must be the same length as the number of output dimensions. If the scores are the outputs
 * of a one v one predictor between all classes then {@link LabelOneVOneTransformer} performs the
 * appropriate scoring operation, and this class will throw an exception when used on such input.
 */
@ProtoSerializableClass(serializedDataClass = LabelTransformerProto.class, version = LabelTransformer.CURRENT_VERSION)
public class LabelTransformer implements OutputTransformer<Label> {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(LabelTransformer.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @Config(description = "Does this transformer produce probabilistic outputs?")
    @ProtoSerializableField
    protected boolean generatesProbabilities = true;

    /**
     * Constructs a LabelTransformer which assumes the model emits probabilities.
     */
    public LabelTransformer() {
        this(true);
    }

    /**
     * Constructs a LabelTransformer.
     * @param generatesProbabilities Does this model emit probabilistic outputs?
     */
    public LabelTransformer(boolean generatesProbabilities) {
        this.generatesProbabilities = generatesProbabilities;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LabelTransformer deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        LabelTransformerProto proto = message.unpack(LabelTransformerProto.class);
        return new LabelTransformer(proto.getGeneratesProbabilities());
    }

    @Override
    public Prediction<Label> transformToPrediction(List<OnnxValue> tensor, ImmutableOutputInfo<Label> outputIDInfo, int numValidFeatures, Example<Label> example) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        if (predictions.length != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, predictions.length = " + predictions.length);
        }
        return generatePrediction(predictions[0],outputIDInfo,numValidFeatures,example);
    }

    private Prediction<Label> generatePrediction(float[] predictions, ImmutableOutputInfo<Label> outputIDInfo, int numUsed, Example<Label> example) {
        Label max = null;
        Map<String,Label> map = new HashMap<>();
        for (int i = 0; i < predictions.length; i++) {
            Label current = new Label(outputIDInfo.getOutput(i).getLabel(),predictions[i]);
            map.put(current.getLabel(),current);
            if ((max == null) || (current.getScore() > max.getScore())) {
                max = current;
            }
        }
        return new Prediction<>(max,map,numUsed,example,generatesProbabilities);
    }

    @Override
    public Label transformToOutput(List<OnnxValue> tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        if (predictions.length != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, predictions.length = " + predictions.length);
        }
        return generateLabel(predictions[0],outputIDInfo);
    }

    private Label generateLabel(float[] predictions, ImmutableOutputInfo<Label> outputIDInfo) {
        int maxIdx = 0;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] > max) {
                maxIdx = i;
                max = predictions[i];
            }
        }
        return new Label(outputIDInfo.getOutput(maxIdx).getLabel(),max);
    }

    /**
     * Rationalises the output of an onnx model into a standard format suitable for
     * downstream work in Tribuo.
     * @param inputs The onnx model output.
     * @param outputIDInfo The output id mapping.
     * @return A 2d array of outputs, the first dimension is batch size, the second dimension is the output space.
     */
    protected float[][] getBatchPredictions(List<OnnxValue> inputs, ImmutableOutputInfo<Label> outputIDInfo) {
        try {
            if (inputs.size() == 1) {
                // Single OnnxTensor [batchSize][numOutputDims]
                if (inputs.get(0) instanceof OnnxTensor) {
                    OnnxTensor output = (OnnxTensor) inputs.get(0);
                    if (output.getInfo().type == OnnxJavaType.FLOAT) {
                        long[] shape = output.getInfo().getShape();
                        if ((shape.length == 2) && (shape[1] == outputIDInfo.size())) {
                            return (float[][]) output.getValue();
                        } else {
                            throw new IllegalArgumentException("Invalid shape for the probabilities tensor, expected shape [batchSize,numOutputs], found " + Arrays.toString(shape));
                        }
                    } else {
                        throw new IllegalArgumentException("Expected the first element to be a float OnnxTensor, found " + inputs.get(0));
                    }
                } else {
                    throw new IllegalArgumentException("Expected the first element to be a float OnnxTensor, found " + inputs.get(0));
                }
            } else if (inputs.size() == 2) {
                // First element is OnnxTensor [batchSize] containing the int predicted label ids, second element is a OnnxSequence<ONNXMap<long,float>>
                if (inputs.get(1) instanceof OnnxSequence) {
                    OnnxSequence seq = (OnnxSequence) inputs.get(1);
                    SequenceInfo info = seq.getInfo();
                    if ((info.sequenceOfMaps) && (info.mapInfo.keyType == OnnxJavaType.INT64) && (info.mapInfo.valueType == OnnxJavaType.FLOAT)) {
                        List<?> output = seq.getValue();
                        float[][] outputArray = new float[output.size()][outputIDInfo.size()];
                        int i = 0;
                        for (Object o : output) {
                            @SuppressWarnings("unchecked") // guarded by the if on the mapInfo above.
                            Map<Long,Float> map = (Map<Long,Float>) o;
                            if (map.size() == outputIDInfo.size()) {
                                for (Map.Entry<Long,Float> e : map.entrySet()) {
                                    Long key = e.getKey();
                                    if (key != (int)(long) key) {
                                        throw new IllegalArgumentException("Key not representable as a Java int, this model is not supported. Expected value less than 2^32, received " + key);
                                    }
                                    outputArray[i][(int)(long)key] = e.getValue();
                                }
                            } else {
                                throw new IllegalArgumentException("Expected " + outputIDInfo.size() + " entries in the " + i + "th element, found " + map.size());
                            }
                            i++;
                        }
                        return outputArray;
                    } else {
                        throw new IllegalArgumentException("Expected a List<Map<Long,Float>>, received a " + info.toString());
                    }
                } else if (inputs.get(1) instanceof OnnxTensor) {
                    OnnxTensor outputScores = (OnnxTensor) inputs.get(1);
                    if (outputScores.getInfo().type == OnnxJavaType.FLOAT) {
                        long[] shape = outputScores.getInfo().getShape();
                        if ((shape.length == 2) && (shape[1] == outputIDInfo.size())) {
                            return (float[][]) outputScores.getValue();
                        } else {
                            throw new IllegalArgumentException("Invalid shape for the score tensor, expected shape [batchSize,numOutputs], found " + Arrays.toString(shape));
                        }
                    } else {
                        throw new IllegalArgumentException("Expected the second element to be a float OnnxTensor, found " + inputs.get(1));
                    }
                } else {
                    throw new IllegalArgumentException("Expected a List<Map<Long,Float>> or a OnnxTensor, received a " + inputs.get(1).getInfo().toString());
                }
            } else {
                throw new IllegalArgumentException("Unexpected number of OnnxValues returned, expected 1 or 2, received " + inputs.size());
            }
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to read a value out of the onnx result.",e);
        }
    }

    @Override
    public List<Prediction<Label>> transformToBatchPrediction(List<OnnxValue> tensor, ImmutableOutputInfo<Label> outputIDInfo, int[] numValidFeatures, List<Example<Label>> examples) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        List<Prediction<Label>> output = new ArrayList<>();

        if ((predictions.length != examples.size()) || (predictions.length != numValidFeatures.length)) {
            throw new IllegalArgumentException("Invalid number of predictions received from the ONNXExternalModel, expected " + numValidFeatures.length + ", received " + predictions.length);
        }

        for (int i = 0; i < predictions.length; i++) {
            output.add(generatePrediction(predictions[i],outputIDInfo,numValidFeatures[i],examples.get(i)));
        }

        return output;
    }

    @Override
    public List<Label> transformToBatchOutput(List<OnnxValue> tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        List<Label> output = new ArrayList<>();

        for (int i = 0; i < predictions.length; i++) {
            output.add(generateLabel(predictions[i],outputIDInfo));
        }

        return output;
    }

    @Override
    public boolean generatesProbabilities() {
        return generatesProbabilities;
    }

    @Override
    public String toString() {
        return "LabelTransformer(generatesProbabilities="+generatesProbabilities+")";
    }

    @Override
    public Class<Label> getTypeWitness() {
        return Label.class;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LabelTransformer that = (LabelTransformer) o;
        return generatesProbabilities == that.generatesProbabilities;
    }

    @Override
    public int hashCode() {
        return Objects.hash(generatesProbabilities);
    }

    @Override
    public OutputTransformerProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OutputTransformer");
    }
}

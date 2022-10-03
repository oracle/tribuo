/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.protos.ProtoSerializableClass;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * Can convert an {@link OnnxValue} into a {@link org.tribuo.Prediction} or a {@link Label}.
 * <p>
 * Accepts:
 * <ul>
 *     <li>a tuple (tensor, float tensor) - as produced by the bare ONNX ML operations (e.g., SVMClassifier).</li>
 *     <li>a single float tensor.</li>
 * </ul>
 * It attempts to parse the output as if it's a vector of predictions from one-v-one classifiers for each class pair.
 * This is the kind of output produced by the ONNX SVMClassifier node, but the ONNX spec is not clear about
 * how this output should be parsed, and ONNX Runtime produces a two element output for binary problems when
 * a strict one-v-one classifier only produces a single output. As a result, this class may need to be updated
 * as ONNX Runtime or the ONNX spec itself evolve.
 * <p>
 * Operates on either a list containing a single tensor [batch_size,(numOutputs*(numOutputs-1))/2], or
 * a list containing two tensors where the second one contains the one-v-one predictions as before.
 */
@ProtoSerializableClass(version = LabelOneVOneTransformer.CURRENT_VERSION)
public final class LabelOneVOneTransformer extends LabelTransformer {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(LabelTransformer.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs a Label transformer that operates on a one v one output and produces scores via voting.
     */
    public LabelOneVOneTransformer() {
        super(false);
    }

    @Override
    public void postConfig() {
        if (generatesProbabilities) {
            throw new PropertyException("", "generatesProbabilities", "generatesProbabilities must not be set to true for this class.");
        }
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static LabelOneVOneTransformer deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new LabelOneVOneTransformer();
    }

    /**
     * Rationalises the output of an onnx model into a standard format suitable for
     * downstream work in Tribuo.
     * <p>
     * It unfolds one-v-one predictions into a score vector using voting. This is used if the
     * model directly outputs the ONNX {@code SVMClassifier} node, as skl2onnx unpacks it for you.
     * <p>
     * Operates on either a list containing a single tensor [batch_size,(numOutputs*(numOutputs-1))/2], or
     * a list containing two tensors where the second one contains the one-v-one predictions as before.
     *
     * @param inputs       The onnx model output.
     * @param outputIDInfo The output id mapping.
     * @return A 2d array of outputs, the first dimension is batch size, the second dimension is the output space.
     */
    @Override
    protected float[][] getBatchPredictions(List<OnnxValue> inputs, ImmutableOutputInfo<Label> outputIDInfo) {
        try {
            if (inputs.size() == 1) {
                // Single OnnxTensor [batchSize][(numOutputs*(numOutputs-1))/2]
                if (inputs.get(0) instanceof OnnxTensor) {
                    OnnxTensor outputScores = (OnnxTensor) inputs.get(0);
                    if (outputScores.getInfo().type == OnnxJavaType.FLOAT) {
                        long[] shape = outputScores.getInfo().getShape();
                        if ((shape.length == 2) && (shape[1] == (outputIDInfo.size() * ((long) outputIDInfo.size() - 1) / 2))) {
                            // Assume the output is one v one and unpack it
                            // Check that the labels and scores are the right shapes
                            // Yes it's annoying LibSVM does it this way.
                            int numOutputs = outputIDInfo.size();
                            float[][] onevone = (float[][]) outputScores.getValue();
                            float[][] scores = new float[(int) shape[0]][numOutputs];
                            for (int k = 0; k < shape[0]; k++) {
                                int counter = 0;
                                for (int i = 0; i < numOutputs; i++) {
                                    for (int j = i + 1; j < numOutputs; j++) {
                                        if (onevone[k][counter] > 0) {
                                            scores[k][i]++;
                                        } else {
                                            scores[k][j]++;
                                        }
                                        counter++;
                                    }
                                }
                            }
                            return scores;
                        } else {
                            throw new IllegalArgumentException("Invalid shape for the score tensor, expected shape [batchSize,(numOutputs*(numOutputs-1))/2], found " + Arrays.toString(shape));
                        }
                    } else {
                        throw new IllegalArgumentException("Expected the first element to be a float OnnxTensor, found " + inputs.get(0));
                    }
                } else {
                    throw new IllegalArgumentException("Expected the first element to be a float OnnxTensor, found " + inputs.get(0));
                }
            } else if (inputs.size() == 2) {
                if (inputs.get(0) instanceof OnnxTensor && inputs.get(1) instanceof OnnxTensor) {
                    OnnxTensor outputLabels = (OnnxTensor) inputs.get(0);
                    OnnxTensor outputScores = (OnnxTensor) inputs.get(1);
                    if (outputScores.getInfo().type == OnnxJavaType.FLOAT) {
                        long[] shape = outputScores.getInfo().getShape();
                        if ((shape.length == 2) && (shape[1] == 2 || (shape[1] == (outputIDInfo.size() * ((long) outputIDInfo.size() - 1) / 2)))) {
                            // Assume the output is one v one and unpack it
                            long[] labelsShape = outputLabels.getInfo().getShape();
                            // Check that the labels and scores are the right shapes
                            if ((labelsShape.length == 1) && (labelsShape[0] == shape[0])) {
                                // Yes it's annoying LibSVM does it this way.
                                int numOutputs = outputIDInfo.size();
                                float[][] onevone = (float[][]) outputScores.getValue();
                                float[][] scores = new float[(int) shape[0]][numOutputs];
                                for (int k = 0; k < shape[0]; k++) {
                                    int counter = 0;
                                    for (int i = 0; i < numOutputs; i++) {
                                        for (int j = i + 1; j < numOutputs; j++) {
                                            if (onevone[k][counter] > 0) {
                                                scores[k][i]++;
                                            } else {
                                                scores[k][j]++;
                                            }
                                            counter++;
                                        }
                                    }
                                }
                                return scores;
                            } else {
                                throw new IllegalArgumentException("Invalid shape for labels, did not match the size of the scores, found labels.shape " + Arrays.toString(labelsShape) + ", and scores.shape " + Arrays.toString(shape));
                            }
                        } else {
                            throw new IllegalArgumentException("Invalid shape for the score tensor, expected shape [batchSize,(numOutputs*(numOutputs-1))/2], found " + Arrays.toString(shape));
                        }
                    } else {
                        throw new IllegalArgumentException("Expected the second element to be a float OnnxTensor, found " + inputs.get(1));
                    }
                } else {
                    throw new IllegalArgumentException("Expected an OnnxTensor, received a " + inputs.get(1).getInfo().toString());
                }
            } else {
                throw new IllegalArgumentException("Unexpected number of OnnxValues returned, expected 2, received " + inputs.size());
            }
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to read a value out of the onnx result.", e);
        }
    }

    @Override
    public String toString() {
        return "LabelOneVOneTransformer(generatesProbabilities="+generatesProbabilities+")";
    }

}

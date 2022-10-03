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

package org.tribuo.interop.oci;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.interop.oci.protos.OCILabelConverterProto;
import org.tribuo.interop.oci.protos.OCIOutputConverterProto;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A converter for {@link DenseMatrix} and {@link DenseVector} into {@link Label} {@link Prediction}s.
 * <p>
 * If the input has length 1 it is assumed to be a single label value, otherwise it
 * must have length = outputIDInfo.size().
 */
@ProtoSerializableClass(serializedDataClass = OCILabelConverterProto.class, version = OCILabelConverter.CURRENT_VERSION)
public final class OCILabelConverter implements OCIOutputConverter<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @ProtoSerializableField
    @Config(mandatory = true, description = "Does this converter produce probabilistic outputs.")
    private boolean generatesProbabilities;

    /**
     * For OLCUT.
     */
    private OCILabelConverter() {}

    /**
     * Constructs an OCILabelConverter with the specified parameters.
     * @param generatesProbabilities Does this converter emit probabilities?
     */
    public OCILabelConverter(boolean generatesProbabilities) {
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
    public static OCILabelConverter deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        OCILabelConverterProto proto = message.unpack(OCILabelConverterProto.class);
        return new OCILabelConverter(proto.getGeneratesProbabilities());
    }

    @Override
    public Prediction<Label> convertOutput(DenseVector scores, int numValidFeature, Example<Label> example, ImmutableOutputInfo<Label> outputIDInfo) {
        if (scores.size() == 1) {
            double labelIdx = scores.get(0);
            if (labelIdx != ((int) labelIdx)) {
                throw new IllegalStateException("Expected a class index, found " + labelIdx);
            }
            Label label = outputIDInfo.getOutput((int) labelIdx);
            if (label != null) {
                return new Prediction<>(label,numValidFeature,example);
            } else {
                throw new IllegalStateException("Expected a class index in the range 0 - " + outputIDInfo.size() + " received " + ((int) labelIdx));
            }
        } else if (scores.size() != outputIDInfo.size()) {
                throw new IllegalStateException("Expected scores for each output, received " + scores.size() + " when there are " + outputIDInfo.size() + "outputs");
        } else {
            Label maxLabel = null;
            Map<String, Label> predMap = new LinkedHashMap<>();
            for (int i = 0; i < scores.size(); i++) {
                String labelName = outputIDInfo.getOutput(i).getLabel();
                double score = scores.get(i);
                Label label = new Label(labelName, score);
                predMap.put(labelName, label);
                if (maxLabel == null || label.getScore() > maxLabel.getScore()) {
                    maxLabel = label;
                }
            }
            return new Prediction<>(maxLabel, predMap, numValidFeature, example, generatesProbabilities);
        }
    }

    @Override
    public List<Prediction<Label>> convertOutput(DenseMatrix scores, int[] numValidFeatures, List<Example<Label>> examples, ImmutableOutputInfo<Label> outputIDInfo) {
        if (scores.getDimension1Size() != examples.size()) {
            throw new IllegalStateException("Expected one prediction per example, recieved " + scores.getDimension1Size() + " predictions when there are " + examples.size() + " examples.");
        }
        List<Prediction<Label>> predictions = new ArrayList<>();
        if (scores.getDimension2Size() == 1) {
            for (int i = 0; i < scores.getDimension1Size(); i++) {
                double labelIdx = scores.get(i,0);
                if (labelIdx != ((int) labelIdx)) {
                    throw new IllegalStateException("Expected a class index at position " + i + ", found " + labelIdx);
                }
                Label label = outputIDInfo.getOutput((int) labelIdx);
                if (label != null) {
                    predictions.add(new Prediction<>(label,numValidFeatures[i],examples.get(i)));
                } else {
                    throw new IllegalStateException("Expected a class index at position " + i + " in the range 0 - " + outputIDInfo.size() + " received " + ((int) labelIdx));
                }
            }
        } else if (scores.getDimension2Size() != outputIDInfo.size()) {
            throw new IllegalStateException("Expected scores for each output, received " + scores.getDimension2Size() + " when there are " + outputIDInfo.size() + "outputs");
        } else {
            for (int i = 0; i < scores.getDimension1Size(); i++) {
                Label maxLabel = null;
                Map<String, Label> predMap = new LinkedHashMap<>();
                for (int j = 0; j < scores.getDimension2Size(); j++) {
                    String labelName = outputIDInfo.getOutput(j).getLabel();
                    double score = scores.get(i, j);
                    Label label = new Label(labelName, score);
                    predMap.put(labelName, label);
                    if (maxLabel == null || label.getScore() > maxLabel.getScore()) {
                        maxLabel = label;
                    }
                }
                predictions.add(new Prediction<>(maxLabel, predMap, numValidFeatures[i], examples.get(i), generatesProbabilities));
            }
        }
        return predictions;
    }

    @Override
    public boolean generatesProbabilities() {
        return generatesProbabilities;
    }

    @Override
    public Class<Label> getTypeWitness() {
        return Label.class;
    }

    @Override
    public String toString() {
        return "OCILabelConverter(generatesProbabilities="+generatesProbabilities+")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OCILabelConverter that = (OCILabelConverter) o;
        return generatesProbabilities == that.generatesProbabilities;
    }

    @Override
    public int hashCode() {
        return Objects.hash(generatesProbabilities);
    }

    @Override
    public OCIOutputConverterProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OCIOutputConverter");
    }
}

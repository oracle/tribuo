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
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto;
import org.tribuo.interop.oci.protos.OCIOutputConverterProto;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;


/**
 * A converter for {@link DenseMatrix} and {@link DenseVector} into {@link MultiLabel} {@link Prediction}s.
 * <p>
 * The threshold is user determined, but defaults to 0.5.
 */
@ProtoSerializableClass(serializedDataClass = OCIMultiLabelConverterProto.class, version = OCIMultiLabelConverter.CURRENT_VERSION)
public final class OCIMultiLabelConverter implements OCIOutputConverter<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * The default threshold for conversion into a label.
     */
    public static final double DEFAULT_THRESHOLD = 0.5;

    @Config(mandatory = true, description = "Does this converter produce probabilistic outputs.")
    private boolean generatesProbabilities;

    @Config(description = "Threshold for generating a label.")
    private double threshold = DEFAULT_THRESHOLD;

    /**
     * For OLCUT.
     */
    private OCIMultiLabelConverter() {}

    /**
     * Constructs an OCILabelConverter with the specified parameters.
     * @param threshold Threshold for emitting a MultiLabel dimension.
     * @param generatesProbabilities Does this converter emit probabilities?
     */
    public OCIMultiLabelConverter(double threshold, boolean generatesProbabilities) {
        this.threshold = threshold;
        this.generatesProbabilities = generatesProbabilities;
        if (generatesProbabilities && (threshold < 0.0 || threshold > 1.0)) {
            throw new IllegalArgumentException("Threshold must be between 0 and 1 to generate probabilities, found " + threshold);
        }
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (generatesProbabilities && (threshold < 0.0 || threshold > 1.0)) {
            throw new PropertyException("","threshold","Threshold must be between 0 and 1 to generate probabilities, found " + threshold);
        }
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static OCIMultiLabelConverter deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        OCIMultiLabelConverterProto proto = message.unpack(OCIMultiLabelConverterProto.class);
        return new OCIMultiLabelConverter(proto.getThreshold(),proto.getGeneratesProbabilities());
    }

    @Override
    public Prediction<MultiLabel> convertOutput(DenseVector scores, int numValidFeature, Example<MultiLabel> example, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        if (scores.size() != outputIDInfo.size()) {
                throw new IllegalStateException("Expected scores for each output, received " + scores.size() + " when there are " + outputIDInfo.size() + "outputs");
        } else {
            Map<String,MultiLabel> fullLabels = new HashMap<>(outputIDInfo.size());
            Set<Label> predictedLabels = new HashSet<>();
            for (int i = 0; i < scores.size(); i++) {
                double labelScore = scores.get(i);
                String labelName = outputIDInfo.getOutput(i).getLabelString();
                Label score = new Label(labelName,labelScore);
                if (labelScore > threshold) {
                    predictedLabels.add(score);
                }
                fullLabels.put(labelName,new MultiLabel(score));
            }
            return new Prediction<>(new MultiLabel(predictedLabels), fullLabels, numValidFeature, example, generatesProbabilities);
        }
    }

    @Override
    public List<Prediction<MultiLabel>> convertOutput(DenseMatrix scores, int[] numValidFeatures, List<Example<MultiLabel>> examples, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        if (scores.getDimension1Size() != examples.size()) {
            throw new IllegalStateException("Expected one prediction per example, recieved " + scores.getDimension1Size() + " predictions when there are " + examples.size() + " examples.");
        }
        if (scores.getDimension2Size() != outputIDInfo.size()) {
            throw new IllegalStateException("Expected scores for each output, received " + scores.getDimension2Size() + " when there are " + outputIDInfo.size() + "outputs");
        } else {
            List<Prediction<MultiLabel>> predictions = new ArrayList<>();
            for (int i = 0; i < scores.getDimension1Size(); i++) {
                Map<String,MultiLabel> fullLabels = new HashMap<>(outputIDInfo.size());
                Set<Label> predictedLabels = new HashSet<>();
                for (int j = 0; j < scores.getDimension2Size(); j++) {
                    double labelScore = scores.get(i,j);
                    String labelName = outputIDInfo.getOutput(j).getLabelString();
                    Label score = new Label(labelName,labelScore);
                    if (labelScore > threshold) {
                        predictedLabels.add(score);
                    }
                    fullLabels.put(labelName,new MultiLabel(score));
                }
                predictions.add(new Prediction<>(new MultiLabel(predictedLabels), fullLabels, numValidFeatures[i], examples.get(i), generatesProbabilities));
            }
            return predictions;
        }
    }

    @Override
    public boolean generatesProbabilities() {
        return generatesProbabilities;
    }

    @Override
    public Class<MultiLabel> getTypeWitness() {
        return MultiLabel.class;
    }

    /**
     * Returns the threshold this converter uses to emit labels.
     * @return The threshold.
     */
    public double getThreshold() {
        return threshold;
    }

    @Override
    public String toString() {
        return "OCIMultiLabelConverter(generatesProbabilities="+generatesProbabilities+")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OCIMultiLabelConverter that = (OCIMultiLabelConverter) o;
        return generatesProbabilities == that.generatesProbabilities && Double.compare(that.threshold, threshold) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(generatesProbabilities, threshold);
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

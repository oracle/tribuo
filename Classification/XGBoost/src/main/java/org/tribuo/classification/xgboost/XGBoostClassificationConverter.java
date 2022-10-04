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

package org.tribuo.classification.xgboost;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.xgboost.XGBoostOutputConverter;
import org.tribuo.common.xgboost.protos.XGBoostOutputConverterProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Converts XGBoost outputs into {@link Label} {@link Prediction}s.
 */
@ProtoSerializableClass(version = XGBoostClassificationConverter.CURRENT_VERSION)
public final class XGBoostClassificationConverter implements XGBoostOutputConverter<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs an XGBoostClassificationConverter.
     */
    public XGBoostClassificationConverter() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static XGBoostClassificationConverter deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new XGBoostClassificationConverter();
    }

    @Override
    public boolean generatesProbabilities() {
        return true;
    }

    @Override
    public Prediction<Label> convertOutput(ImmutableOutputInfo<Label> info, List<float[]> probabilitiesList, int numValidFeatures, Example<Label> example) {
        if (probabilitiesList.size() != 1) {
            throw new IllegalArgumentException("XGBoostClassificationConverter only expects a single model output.");
        }
        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        LinkedHashMap<String,Label> probMap = new LinkedHashMap<>();
        float[] probabilities = probabilitiesList.get(0);

        for (int i = 0; i < probabilities.length; i++) {
            String name = info.getOutput(i).getLabel();
            Label label = new Label(name, probabilities[i]);
            probMap.put(name, label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }

        return new Prediction<>(maxLabel,probMap,numValidFeatures,example,true);
    }

    @Override
    public List<Prediction<Label>> convertBatchOutput(ImmutableOutputInfo<Label> info, List<float[][]> probabilitiesList, int[] numValidFeatures, Example<Label>[] examples) {
        if (probabilitiesList.size() != 1) {
            throw new IllegalArgumentException("XGBoostClassificationConverter only expects a single model output.");
        }
        float[][] probabilities = probabilitiesList.get(0);

        List<Prediction<Label>> predictions = new ArrayList<>();
        for (int i = 0; i < probabilities.length; i++) {
            double maxScore = Double.NEGATIVE_INFINITY;
            Label maxLabel = null;
            LinkedHashMap<String, Label> probMap = new LinkedHashMap<>();
            for (int j = 0; j < probabilities[i].length; j++) {
                String name = info.getOutput(j).getLabel();
                Label label = new Label(name, probabilities[i][j]);
                probMap.put(name, label);
                if (label.getScore() > maxScore) {
                    maxScore = label.getScore();
                    maxLabel = label;
                }
            }

            predictions.add(new Prediction<>(maxLabel, probMap, numValidFeatures[i], examples[i], true));
        }

        return predictions;
    }

    @Override
    public XGBoostOutputConverterProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public Class<Label> getTypeWitness() {
        return Label.class;
    }
}

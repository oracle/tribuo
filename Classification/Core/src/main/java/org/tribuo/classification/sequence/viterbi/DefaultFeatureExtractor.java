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

package org.tribuo.classification.sequence.viterbi;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Feature;
import org.tribuo.classification.Label;
import org.tribuo.classification.protos.DefaultFeatureExtractorProto;
import org.tribuo.classification.protos.LabelFeatureExtractorProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A label feature extractor that produces several kinds of label-based features.
 * <p>
 * The options are: the most recent output, the least recent output, recent bigrams, recent trigrams, recent 4-grams.
 */
@ProtoSerializableClass(serializedDataClass = DefaultFeatureExtractorProto.class, version = DefaultFeatureExtractor.CURRENT_VERSION)
public class DefaultFeatureExtractor implements LabelFeatureExtractor {

    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * indicates the position of the first (most recent) outcome to include. For example, the
     * default value of 1 means that if the outcomes produced so far by the classifier were [A, B,
     * C, D], then the first outcome to be used as a feature would be D since it is the most recent.
     */
    @Config(mandatory = true, description = "Position of the most recent outcome to include.")
    @ProtoSerializableField
    private int mostRecentOutcome;

    /**
     * indicates the position of the last (least recent) outcome to include. For example, the
     * default value of 3 means that if the outcomes produced so far by the classifier were [A, B,
     * C, D], then the last outcome to be used as a feature would be B since and is considered the
     * least recent.
     */
    @Config(mandatory = true, description = "Position of the least recent output to include.")
    @ProtoSerializableField
    private int leastRecentOutcome;

    /**
     * when true indicates that bigrams of outcomes should be included as features
     */
    @Config(mandatory = true, description = "Use bigrams of the labels as features.")
    @ProtoSerializableField
    private boolean useBigram;

    /**
     * indicates that trigrams of outcomes should be included as features
     */
    @Config(mandatory = true, description = "Use trigrams of the labels as features.")
    @ProtoSerializableField
    private boolean useTrigram;

    /**
     * indicates that 4-grams of outcomes should be included as features
     */
    @Config(mandatory = true, description = "Use 4-grams of the labels as features.")
    @ProtoSerializableField
    private boolean use4gram;

    /**
     * Constructs a default feature extractor for bigrams and trigrams using the past 3 outcomes.
     */
    public DefaultFeatureExtractor() {
        this(1, 3, true, true, false);
    }

    /**
     * Constructs a default feature extractor using the supplied parameters.
     * @param mostRecentOutcome The most recent outcome to include as a feature.
     * @param leastRecentOutcome The least recent outcome to include as a feature.
     * @param useBigram Use bigrams of the outcomes.
     * @param useTrigram Use trigrams of the outcomes.
     * @param use4gram Use 4-grams of the outcomes.
     */
    public DefaultFeatureExtractor(int mostRecentOutcome, int leastRecentOutcome, boolean useBigram, boolean useTrigram, boolean use4gram) {
        this.mostRecentOutcome = mostRecentOutcome;
        this.leastRecentOutcome = leastRecentOutcome;
        this.useBigram = useBigram;
        this.useTrigram = useTrigram;
        this.use4gram = use4gram;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static DefaultFeatureExtractor deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        DefaultFeatureExtractorProto proto = message.unpack(DefaultFeatureExtractorProto.class);
        return new DefaultFeatureExtractor(proto.getMostRecentOutcome(), proto.getLeastRecentOutcome(),
            proto.getUseBigram(), proto.getUseTrigram(), proto.getUse4Gram());
    }

    @Override
    public String toString() {
        return "DefaultFeatureExtractor(mostRecent=" + mostRecentOutcome + ",leastRecent=" + leastRecentOutcome + ",useBigram=" + useBigram + ",useTrigram=" + useTrigram + ",use4gram=" + use4gram + ")";
    }

    @Override
    public List<Feature> extractFeatures(List<Label> previousOutcomes, double value) {
        if (previousOutcomes == null || previousOutcomes.size() == 0) {
            return Collections.emptyList();
        }

        List<Feature> features = new ArrayList<>();

        for (int i = mostRecentOutcome; i <= leastRecentOutcome; i++) {
            int index = previousOutcomes.size() - i;
            if (index >= 0) {
                Feature feature = new Feature("PreviousOutcome_L" + i + "_" + previousOutcomes.get(index).getLabel(), value);
                features.add(feature);
            }
        }

        if (useBigram && previousOutcomes.size() >= 2) {
            int size = previousOutcomes.size();
            String featureValue = previousOutcomes.get(size - 1).getLabel() + "_" + previousOutcomes.get(size - 2).getLabel();
            Feature feature = new Feature("PreviousOutcomes_L1_2gram_L2R_" + featureValue, value);
            features.add(feature);
        }

        if (useTrigram && previousOutcomes.size() >= 3) {
            int size = previousOutcomes.size();
            String featureValue = previousOutcomes.get(size - 1).getLabel() + "_" + previousOutcomes.get(size - 2).getLabel() + "_"
                    + previousOutcomes.get(size - 3).getLabel();
            Feature feature = new Feature("PreviousOutcomes_L1_3gram_L2R_" + featureValue, value);
            features.add(feature);
        }

        if (use4gram && previousOutcomes.size() >= 4) {
            int size = previousOutcomes.size();
            String featureValue = previousOutcomes.get(size - 1).getLabel() + "_" + previousOutcomes.get(size - 2).getLabel() + "_"
                    + previousOutcomes.get(size - 3).getLabel() + "_" + previousOutcomes.get(size - 4).getLabel();
            Feature feature = new Feature("PreviousOutcomes_L1_4gram_L2R_" + featureValue, value);
            features.add(feature);
        }

        return features;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "LabelFeatureExtractor");
    }

    @Override
    public LabelFeatureExtractorProto serialize() {
        return ProtoUtil.serialize(this);
    }
}

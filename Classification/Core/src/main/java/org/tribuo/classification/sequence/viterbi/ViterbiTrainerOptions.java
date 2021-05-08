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

package org.tribuo.classification.sequence.viterbi;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;

/**
 * Options for building a viterbi trainer.
 */
public class ViterbiTrainerOptions implements Options {

    /**
     * Type of label features to include.
     */
    public enum ViterbiLabelFeatures {
        /**
         * The default label features.
         */
        DEFAULT,
        /**
         * No label features.
         */
        NONE
    }

    @Option(longName = "viterbi-score-aggregation", usage = "Aggregation operation, choices are {ADD, MULTIPLY}.")
    private ViterbiModel.ScoreAggregation viterbiScoreAggregation = ViterbiModel.ScoreAggregation.ADD;

    @Option(longName = "viterbi-label-features", usage = "Add label features to the inner training, choices are {DEFAULT, NONE}.")
    private ViterbiLabelFeatures viterbiLabelFeatures = ViterbiLabelFeatures.DEFAULT;

    @Option(longName = "viterbi-stack-size", usage = "-1 for no limit on the stack size")
    private int viterbiStackSize = -1;

    /**
     * Creates a viterbi trainer wrapping the supplied label trainer.
     * @param innerTrainer The trainer to wrap.
     * @return A ViterbiTrainer.
     */
    public ViterbiTrainer getSequenceTrainer(Trainer<Label> innerTrainer) {
        LabelFeatureExtractor lfe = new NoopFeatureExtractor();
        if (viterbiLabelFeatures == ViterbiLabelFeatures.DEFAULT) {
            lfe = new DefaultFeatureExtractor();
        }
        return new ViterbiTrainer(innerTrainer, lfe, viterbiStackSize, viterbiScoreAggregation);
    }
}

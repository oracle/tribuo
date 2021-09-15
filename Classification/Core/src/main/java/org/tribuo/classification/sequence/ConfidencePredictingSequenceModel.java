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

package org.tribuo.classification.sequence;

import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.sequence.SequenceExample;
import org.tribuo.sequence.SequenceModel;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * A Sequence model which can provide confidence predictions for subsequence predictions.
 * <p>
 * Used to provide confidence scores on a per subsequence level.
 * <p>
 * The exemplar of this is providing a confidence score for each Named Entity present
 * in a SequenceExample.
 */
public abstract class ConfidencePredictingSequenceModel extends SequenceModel<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * Constructs a ConfidencePredictingSequenceModel with the supplied parameters.
     * @param name The model name.
     * @param description The model provenance.
     * @param featureIDMap The feature domain.
     * @param labelIDMap The output domain.
     */
    protected ConfidencePredictingSequenceModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap) {
        super(name,description,featureIDMap,labelIDMap);
    }

    /**
     * The scoring function for the subsequences. Provides the scores which should be assigned to each subsequence.
     * @param example The input sequence example.
     * @param predictions The predictions produced by this model.
     * @param subsequences The subsequences to score.
     * @param <SUB> The subsequence type.
     * @return The scores for the subsequences.
     */
    public abstract <SUB extends Subsequence> List<Double> scoreSubsequences(SequenceExample<Label> example, List<Prediction<Label>> predictions, List<SUB> subsequences);

    /**
     * A scoring method which multiplies together the per prediction scores.
     * @param predictions The element level predictions.
     * @param subsequences The subsequences denoting prediction boundaries.
     * @param <SUB> The subsequence type.
     * @return A list of scores for each subsequence.
     */
    public static <SUB extends Subsequence> List<Double> multiplyWeights(List<Prediction<Label>> predictions, List<SUB> subsequences) {
        List<Double> scores = new ArrayList<>(subsequences.size());
        for(Subsequence subsequence : subsequences) {
            scores.add(multiplyWeights(predictions, subsequence));
        }
        return scores;
    }

    private static <SUB extends Subsequence> Double multiplyWeights(List<Prediction<Label>> predictions, SUB subsequence) {
        double counter = 1.0;
        for (int i=subsequence.begin; i<subsequence.end; i++) {
            counter *= predictions.get(i).getOutput().getScore();
        }
        return counter;
    }

    /**
     * A range class used to define a subsequence of a SequenceExample.
     */
    public static class Subsequence implements Serializable {
        private static final long serialVersionUID = 1L;
        /**
         * The subsequence start index.
         */
        public final int begin;
        /**
         * The subsequence end index.
         */
        public final int end;

        /**
         * Constructs a subsequence for the fixed range, exclusive of the end.
         * @param begin The start element.
         * @param end The end element.
         */
        public Subsequence(int begin, int end) {
            this.begin = begin;
            this.end = end;
        }

        /**
         * Returns the number of elements in this subsequence.
         * @return The length of the subsequence.
         */
        public int length() {
            return end - begin;
        }

        @Override
        public String toString() {
            return "("+begin+","+end+")";
        }
    }

}

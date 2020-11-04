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

package org.tribuo.anomaly.evaluation;

import org.tribuo.anomaly.Event;
import org.tribuo.evaluation.Evaluation;

/**
 * An {@link Evaluation} for anomaly detection {@link Event}s.
 */
public interface AnomalyEvaluation extends Evaluation<Event> {

    /**
     * Returns the number of false positives, i.e., expected events classified as anomalous.
     * @return The number of false positives.
     */
    long getFalsePositives();

    /**
     * Returns the number of true positives, i.e., anomalous events classified as anomalous.
     * @return The number of true positives.
     */
    long getTruePositives();

    /**
     * Returns the number of true negatives, i.e., expected events classified as events.
     * @return The number of true negatives.
     */
    long getTrueNegatives();

    /**
     * Returns the number of false negatives, i.e., anomalous events classified as expected.
     * <p>
     * These are the ones you don't want.
     * @return The number of false negatives.
     */
    long getFalseNegatives();

    /**
     * Returns the precision of the anomalous events, i.e., true positives divided by the number of predicted positives.
     * @return The precision.
     */
    double getPrecision();

    /**
     * Returns the recall of the anomalous events, i.e., true positives divided by the number of positives.
     * @return The recall.
     */
    double getRecall();

    /**
     * Returns the F_1 score of the anomalous events, i.e., the harmonic mean of the precision and the recall.
     * @return The F_1 score.
     */
    double getF1();

    /**
     * Returns a confusion matrix formatted String for display.
     * @return The confusion matrix in a String.
     */
    String confusionString();

}
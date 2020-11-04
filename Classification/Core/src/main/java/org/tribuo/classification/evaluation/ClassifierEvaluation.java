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

package org.tribuo.classification.evaluation;

import org.tribuo.classification.Classifiable;
import org.tribuo.evaluation.Evaluation;

/**
 * Defines methods that calculate classification performance, used for both multi-class and multi-label classification.
 *
 * @param <T> The output type.
 */
public interface ClassifierEvaluation<T extends Classifiable<T>> extends Evaluation<T> {

    /**
     * Returns the number of times label {@code truth} was predicted as label {@code predicted}.
     *
     * @param predicted The predicted label.
     * @param truth     The true label.
     * @return The number of times the predicted label was returned for the true label.
     */
    public double confusion(T predicted, T truth);

    // TODO add support(), support(label)

    /**
     * Returns the number of true positives, i.e., the number of times the label was correctly predicted.
     *
     * @param label The label to calculate.
     * @return The number of true positives for that label.
     */
    public double tp(T label);

    /**
     * Returns the micro average of the number of true positives across all the labels, i.e., the total
     * number of true positives.
     *
     * @return The micro averaged number of true positives.
     */
    public double tp();

    /**
     * Returns the macro averaged number of true positives, averaged across the labels.
     *
     * @return The macro averaged number of true positives.
     */
    public double macroTP();

    /**
     * Returns the number of false positives, i.e., the number of times this label was predicted but it was not the true label..
     *
     * @param label the label to calculate.
     * @return The number of false positives for that label.
     */
    public double fp(T label);

    /**
     * Returns the micro average of the number of false positives across all the labels, i.e., the total
     * number of false positives.
     *
     * @return The micro averaged number of false positives.
     */
    public double fp();

    /**
     * Returns the macro averaged number of false positives, averaged across the labels.
     *
     * @return The macro averaged number of false positives.
     */
    public double macroFP();

    /**
     * Returns the number of true negatives for that label, i.e., the number of times it wasn't predicted, and was not the true label.
     *
     * @param label The label to use.
     * @return the number of true negatives.
     */
    public double tn(T label);

    /**
     * Returns the total number of true negatives. This isn't very useful in multiclass problems.
     *
     * @return The number of true negatives.
     */
    public double tn();

    /**
     * Returns the macro averaged number of true negatives.
     *
     * @return The macro averaged number of true negatives.
     */
    public double macroTN();

    /**
     * Returns the number of false negatives, i.e., the number of times the true label was incorrectly predicted as another label.
     *
     * @param label The true label.
     * @return The number of false negatives.
     */
    public double fn(T label);

    /**
     * Returns the micro averaged number of false negatives.
     *
     * @return The micro averaged number of false negatives.
     */
    public double fn();

    /**
     * Returns the macro averaged number of false negatives.
     *
     * @return The macro averaged number of false negatives.
     */
    public double macroFN();

    /**
     * Returns the precision of this label, i.e., the number of true positives divided by the number of true positives plus false positives.
     *
     * @param label The label.
     * @return The precision.
     */
    public double precision(T label);

    /**
     * Returns the micro averaged precision.
     *
     * @return The micro averaged precision.
     */
    public double microAveragedPrecision();

    /**
     * Returns the macro averaged precision.
     *
     * @return The macro averaged precision.
     */
    public double macroAveragedPrecision();

    /**
     * Returns the recall of this label, i.e., the number of true positives divided by the number of true positives plus false negatives.
     *
     * @param label The label.
     * @return The recall.
     */
    public double recall(T label);

    /**
     * Returns the micro averaged recall.
     *
     * @return The micro averaged recall.
     */
    public double microAveragedRecall();

    /**
     * Returns the macro averaged recall.
     *
     * @return The macro averaged recall.
     */
    public double macroAveragedRecall();

    /**
     * Returns the F_1 score, i.e., the harmonic mean of the precision and recall.
     *
     * @param label The label.
     * @return The F_1 score.
     */
    public double f1(T label);

    /**
     * Returns the micro averaged F_1 across all labels.
     *
     * @return The F_1 score.
     */
    public double microAveragedF1();

    /**
     * Returns the macro averaged F_1 across all the labels.
     *
     * @return The F_1 score.
     */
    public double macroAveragedF1();

    /**
     * Returns the balanced error rate, i.e., the mean of the per label recalls.
     *
     * @return The balanced error rate.
     */
    public double balancedErrorRate();

    /**
     * Returns the underlying confusion matrix.
     * @return The confusion matrix.
     */
    public ConfusionMatrix<T> getConfusionMatrix();
}

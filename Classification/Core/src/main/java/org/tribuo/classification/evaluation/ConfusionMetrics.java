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
import org.tribuo.evaluation.metrics.EvaluationMetric.Average;
import org.tribuo.evaluation.metrics.MetricTarget;

import java.util.logging.Logger;

/**
 * Static functions for computing classification metrics based on a {@link ConfusionMatrix}.
 */
public final class ConfusionMetrics {

    private static final Logger logger = Logger.getLogger(ConfusionMetrics.class.getName());

    // singleton
    private ConfusionMetrics() { }

    /**
     * Calculates the accuracy given this confusion matrix.
     *
     * @param <T>    The type parameter
     * @param target The metric target
     * @param cm     The confusion matrix
     * @return The accuracy
     */
    public static <T extends Classifiable<T>> double accuracy(MetricTarget<T> target, ConfusionMatrix<T> cm) {
        if (target.getOutputTarget().isPresent()) {
            return accuracy(target.getOutputTarget().get(), cm);
        } else {
            return accuracy(target.getAverageTarget().get(), cm);
        }
    }

    /**
     * Calculates a per label accuracy given this confusion matrix.
     *
     * @param <T>   The type parameter
     * @param label The label
     * @param cm    The confusion matrix
     * @return The accuracy
     */
    public static <T extends Classifiable<T>> double accuracy(T label, ConfusionMatrix<T> cm) {
        double support = cm.support(label);
        // handle div-by-zero
        if (support == 0d) {
            logger.warning("No predictions for " + label + ": accuracy ill-defined");
            return Double.NaN;
        }
        return cm.tp(label) / cm.support(label);
    }

    /**
     * Calculates the accuracy using the specified average type and confusion matrix.
     *
     * @param <T>     the type parameter
     * @param average the average
     * @param cm      The confusion matrix
     * @return The accuracy
     */
    public static <T extends Classifiable<T>> double accuracy(Average average, ConfusionMatrix<T> cm) {
        if (average.equals(Average.MICRO)) {
            // handle div-by-zero
            if (cm.support() == 0d) {
                logger.warning("No predictions: accuracy ill-defined");
                return Double.NaN;
            }
            return cm.tp() / cm.support();
        } else {
            // handle div-by-zero
            if (cm.getDomain().size() == 0) {
                logger.warning("Empty domain: accuracy ill-defined");
                return Double.NaN;
            }
            double total = 0d;
            for (T output : cm.getDomain().getDomain()) {
                total += accuracy(output, cm);
            }
            return total / cm.getDomain().size();
        }
    }

    /**
     * Calculates the balanced error rate, i.e., the mean of the recalls.
     *
     * @param <T> the type parameter
     * @param cm  The confusion matrix
     * @return the balanced error rate.
     */
    public static <T extends Classifiable<T>> double balancedErrorRate(ConfusionMatrix<T> cm) {
        // handle div-by-zero
        if (cm.getDomain().size() == 0) {
            logger.warning("Empty domain: balanced error rate ill-defined");
            return Double.NaN;
        }
        double sr = 0d;
        for (T output : cm.getDomain().getDomain()) {
            sr += recall(new MetricTarget<>(output), cm);
        }
        return 1d - (sr / cm.getDomain().size());
    }

    /**
     * Computes the confusion function value for a given metric target and confusion matrix.
     * <p>
     * For example - to compute macro precision:
     *
     * <code>
     * ConfusionFunction&lt;T&gt; fxn = ConfusionMetric::precision;
     * MetricTarget&lt;T&gt; tgt = new MetricTarget(Average.macro)
     * ConfusionMatrix&lt;T&gt; cm = ...
     * compute(fxn, tgt, cm);
     * </code>
     * <p>
     * This is equivalent to the following:
     *
     * <code>
     * ConfusionMatrix&lt;T&gt; cm = ...
     * double total = 0d;
     * for (T label : cm.getDomain().getDomain()) {
     * total += precision(cm.tp(label), cm.tp(label), ...);
     * }
     * double avg = total / cm.getDomain().size()
     * </code>
     *
     * @param fxn the confusion function
     * @param tgt the metric target
     * @param cm  the confusion matrix
     * @param <T> the output type
     * @return the value of fxn applied to (tgt, cm)
     */
    private static <T extends Classifiable<T>> double compute(ConfusionFunction<T> fxn, MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return fxn.compute(tgt, cm);
    }

    /**
     * Returns the number of true positives, possibly averaged depending on the metric target.
     *
     * @param <T> the type parameter
     * @param tgt The metric target
     * @param cm  The confusion matrix
     * @return the true positives.
     */
    public static <T extends Classifiable<T>> double tp(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return compute(ConfusionMetrics::tp, tgt, cm);
    }

    /**
     * Returns the number of false positives, possibly averaged depending on the metric target.
     *
     * @param <T> the type parameter
     * @param tgt The metric target
     * @param cm  The confusion matrix
     * @return the false positives.
     */
    public static <T extends Classifiable<T>> double fp(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return compute(ConfusionMetrics::fp, tgt, cm);
    }

    /**
     * Returns the number of true negatives, possibly averaged depending on the metric target.
     *
     * @param <T> the type parameter
     * @param tgt The metric target
     * @param cm  The confusion matrix
     * @return the true negatives.
     */
    public static <T extends Classifiable<T>> double tn(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return compute(ConfusionMetrics::tn, tgt, cm);
    }

    /**
     * Returns the number of false negatives, possibly averaged depending on the metric target.
     *
     * @param <T> the type parameter
     * @param tgt The metric target
     * @param cm  The confusion matrix
     * @return the false negatives.
     */
    public static <T extends Classifiable<T>> double fn(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return compute(ConfusionMetrics::fn, tgt, cm);
    }

    /**
     * Helper function to return the specified argument. Used as a method reference.
     * @param tp The true positives.
     * @param fp The false positives.
     * @param tn The true negatives.
     * @param fn The false negatives.
     * @return The true positives.
     */
    private static double tp(double tp, double fp, double tn, double fn) {
        return tp;
    }

    /**
     * Helper function to return the specified argument. Used as a method reference.
     * @param tp The true positives.
     * @param fp The false positives.
     * @param tn The true negatives.
     * @param fn The false negatives.
     * @return The false positives.
     */
    private static double fp(double tp, double fp, double tn, double fn) {
        return fp;
    }

    /**
     * Helper function to return the specified argument. Used as a method reference.
     * @param tp The true positives.
     * @param fp The false positives.
     * @param tn The true negatives.
     * @param fn The false negatives.
     * @return The true negatives.
     */
    private static double tn(double tp, double fp, double tn, double fn) {
        return tn;
    }

    /**
     * Helper function to return the specified argument. Used as a method reference.
     * @param tp The true positives.
     * @param fp The false positives.
     * @param tn The true negatives.
     * @param fn The false negatives.
     * @return The false negatives.
     */
    private static double fn(double tp, double fp, double tn, double fn) {
        return fn;
    }

    //
    // PRECISION ---------------------------------------------------------------
    //

    /**
     * Calculates the precision for this metric target.
     *
     * @param <T> the type parameter
     * @param tgt The metric target
     * @param cm  The confusion matrix
     * @return the precision.
     */
    public static <T extends Classifiable<T>> double precision(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return compute(ConfusionMetrics::precision, tgt, cm);
    }

    /**
     * Calculates the precision based upon the supplied statistics.
     *
     * @param tp  the true positives
     * @param fp  the false positives
     * @param tn  the true negatives
     * @param fn  the false negatives
     * @return The recall.
     */
    public static double precision(double tp, double fp, double tn, double fn) {
        double denom = tp + fp;
        // If the denominator is 0, return 0 (as opposed to Double.NaN, say)
        return (denom == 0) ? 0d : tp / denom;
    }

    //
    // RECALL ------------------------------------------------------------------
    //

    /**
     * Calculates the recall for this metric target.
     *
     * @param <T> the type parameter
     * @param tgt The metric target
     * @param cm  The confusion matrix
     * @return The recall.
     */
    public static <T extends Classifiable<T>> double recall(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return compute(ConfusionMetrics::recall, tgt, cm);
    }

    /**
     * Calculates the recall based upon the supplied statistics.
     *
     * @param tp  the true positives
     * @param fp  the false positives
     * @param tn  the true negatives
     * @param fn  the false negatives
     * @return The recall.
     */
    public static double recall(double tp, double fp, double tn, double fn) {
        double denom = tp + fn;
        // If the denominator is 0, return 0 (as opposed to Double.NaN, say)
        return (denom == 0) ? 0d : tp / denom;
    }

    //
    // F-SCORE -----------------------------------------------------------------
    //

    /**
     * Computes the F_1 score.
     *
     * @param <T> the type parameter
     * @param tgt the metric target.
     * @param cm  the confusion matrix.
     * @return the F_1 score.
     */
    public static <T extends Classifiable<T>> double f1(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
        return compute(ConfusionMetrics::f1, tgt, cm);
    }

    /**
     * Computes the F_1 score.
     *
     * @param tp  the true positives
     * @param fp  the false positives
     * @param tn  the true negatives
     * @param fn  the false negatives
     * @return the F_1 score.
     */
    public static double f1(double tp, double fp, double tn, double fn) {
        return fscore(1d, tp, fp, tn, fn);
    }

    /**
     * Computes the Fscore.
     *
     * @param beta the beta.
     * @param tp   the true positives.
     * @param fp   the false positives.
     * @param tn   the true negatives.
     * @param fn   the false negatives.
     * @return the F_beta score.
     */
    public static double fscore(double beta, double tp, double fp, double tn, double fn) {
        double bsq = beta * beta;
        double p = precision(tp, fp, tn, fn);
        double r = recall(tp, fp, tn, fn);
        double denom = (bsq * p) + r;
        return (denom == 0) ? 0d : (1 + bsq) * p * r / denom;
    }

    /**
     * Computes the Fscore.
     *
     * @param <T>  the type parameter
     * @param tgt  The metric target
     * @param cm   The confusion matrix
     * @param beta the beta
     * @return The F_beta score.
     */
    public static <T extends Classifiable<T>> double fscore(MetricTarget<T> tgt, ConfusionMatrix<T> cm, double beta) {
        ConfusionFunction<T> fxn = (tp, fp, tn, fn) -> fscore(beta, tp, fp, tn, fn);
        return compute(fxn, tgt, cm);
    }

    /**
     * A function that takes a {@link MetricTarget} and {@link ConfusionMatrix} as inputs and outputs the value of
     * the confusion metric specified in the implementation of
     * {@link ConfusionFunction#compute(double, double, double, double)}.
     *
     * @param <T> The classification type.
     */
    @FunctionalInterface
    private static interface ConfusionFunction<T extends Classifiable<T>> {

        /**
         * Provides a uniform function signature for a bunch of different metrics.
         *
         * @param tp the true positives.
         * @param fp the false positives.
         * @param tn the true negatives.
         * @param fn the false negatives.
         * @return the value.
         */
        double compute(double tp, double fp, double tn, double fn);

        /**
         * Compute the value.
         *
         * @param tgt the metric target.
         * @param cm  the confusion matrix.
         * @return the value.
         */
        default double compute(MetricTarget<T> tgt, ConfusionMatrix<T> cm) {
            if (tgt.getOutputTarget().isPresent()) {
                return compute(tgt.getOutputTarget().get(), cm);
            } else if (tgt.getAverageTarget().isPresent()) {
                return compute(tgt.getAverageTarget().get(), cm);
            } else {
                throw new IllegalStateException("MetricTarget with no actual target");
            }
        }

        /**
         * Compute the value.
         *
         * @param label the target label.
         * @param cm    the confusion matrix.
         * @return the value.
         */
        default double compute(T label, ConfusionMatrix<T> cm) {
            return compute(cm.tp(label), cm.fp(label), cm.tn(label), cm.fn(label));
        }

        /**
         * Compute the value.
         *
         * @param average the average type.
         * @param cm      the confusion matrix.
         * @return the value.
         */
        default double compute(Average average, ConfusionMatrix<T> cm) {
            switch (average) {
                case MACRO:
                    if (cm.getDomain().size() == 0) {
                        logger.warning("Empty domain: macro-average ill-defined.");
                        return Double.NaN;
                    }
                    double total = 0d;
                    for (T output : cm.getDomain().getDomain()) {
                        total += compute(output, cm);
                    }
                    return total / cm.getDomain().size();
                case MICRO:
                    if (cm.support() == 0) {
                        logger.warning("No predictions: micro-average ill-defined.");
                        return Double.NaN;
                    }
                    return compute(cm.tp(), cm.fp(), cm.tn(), cm.fn());
                default:
                    throw new IllegalArgumentException("Unsupported average type: " + average.name());
            }
        }
    }

}
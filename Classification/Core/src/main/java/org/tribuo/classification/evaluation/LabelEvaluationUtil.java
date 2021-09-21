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

import com.oracle.labs.mlrg.olcut.util.SortUtil;
import org.tribuo.classification.Label;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Static utility functions for calculating performance metrics on {@link Label}s.
 */
public final class LabelEvaluationUtil {

    // Static utility class, has private constructor and is final.
    private LabelEvaluationUtil() {}

    /**
     * Summarises a Precision-Recall Curve by taking the weighted mean of the
     * precisions at a given threshold, where the weight is the recall achieved at
     * that threshold.
     *
     * Follows scikit-learn's implementation.
     *
     * In general use the AUC for a Precision-Recall Gain curve as the area under
     * the precision-recall curve is not properly normalized.
     * @param yPos Each element is true if the label was from the positive class.
     * @param yScore Each element is the score of the positive class.
     * @return The averaged precision.
     */
    public static double averagedPrecision(boolean[] yPos, double[] yScore) {
        PRCurve prc = generatePRCurve(yPos,yScore);

        double score = 0.0;

        for (int i = 0; i < prc.precision.length-1; i++) {
            score += (prc.recall[i+1] - prc.recall[i]) * prc.precision[i];
        }

        return -score;
    }

    /**
     * Calculates the Precision Recall curve for a single label.
     *
     * In general use Precision-Recall Gain curves.
     * @param yPos Each element is true if the label was from the positive class.
     * @param yScore Each element is the score of the positive class.
     * @return The PRCurve for one label.
     */
    public static PRCurve generatePRCurve(boolean[] yPos, double[] yScore) {
        TPFP tpfp = generateTPFPs(yPos,yScore);

        ArrayList<Double> precisions = new ArrayList<>(tpfp.falsePos.size());
        ArrayList<Double> recalls = new ArrayList<>(tpfp.falsePos.size());
        ArrayList<Double> thresholds = new ArrayList<>(tpfp.falsePos.size());

        for (int i = 0; i < tpfp.falsePos.size(); i++) {
            double curFalsePos = tpfp.falsePos.get(i);
            double curTruePos = tpfp.truePos.get(i);

            double precision = 0.0;
            double recall = 0.0;
            if (curTruePos != 0) {
                precision = curTruePos / (curTruePos + curFalsePos);
                recall = curTruePos / tpfp.totalPos;
            }

            precisions.add(precision);
            recalls.add(recall);
            thresholds.add(tpfp.thresholds.get(i));

            // Break out if we've achieved full recall.
            if (curTruePos == tpfp.totalPos) {
                break;
            }
        }

        Collections.reverse(precisions);
        Collections.reverse(recalls);
        Collections.reverse(thresholds);

        precisions.add(1.0);
        recalls.add(0.0);

        return new PRCurve(Util.toPrimitiveDouble(precisions),Util.toPrimitiveDouble(recalls),Util.toPrimitiveDouble(thresholds));
    }

    /**
     * Calculates the area under the receiver operator characteristic curve,
     * i.e., the AUC of the ROC curve.
     * @param yPos Is the associated index a positive label.
     * @param yScore The score of the positive class.
     * @return The auc (a value bounded 0.0-1.0).
     */
    public static double binaryAUCROC(boolean[] yPos, double[] yScore) {
        ROC roc = generateROCCurve(yPos, yScore);
        return Util.auc(roc.fpr, roc.tpr);
    }

    /**
     * Calculates the binary ROC for a single label.
     * @param yPos Each element is true if the label was from the positive class.
     * @param yScore Each element is the score of the positive class.
     * @return The ROC for one label.
     */
    public static ROC generateROCCurve(boolean[] yPos, double[] yScore) {
        TPFP tpfp = generateTPFPs(yPos,yScore);

        // If it doesn't exist, add a 0,0 point so the graph always starts from the origin.
        // This point has a threshold of POSITIVE_INFINITY as it's the always negative classifier.
        if ((tpfp.truePos.get(0) != 0) || (tpfp.falsePos.get(0) != 0)) {
            tpfp.truePos.add(0,0);
            tpfp.falsePos.add(0,0);
            tpfp.thresholds.add(0,Double.POSITIVE_INFINITY); // Set threshold to positive infinity
        }

        // Transform things back into arrays.
        double[] truePosArr = Util.toPrimitiveDoubleFromInteger(tpfp.truePos);
        double[] falsePosArr = Util.toPrimitiveDoubleFromInteger(tpfp.falsePos);
        double[] thresholdsArr = Util.toPrimitiveDouble(tpfp.thresholds);

        // Convert from counts into a rate.
        double maxTrue = truePosArr[truePosArr.length-1];
        double maxFalse = falsePosArr[falsePosArr.length-1];
        for (int i = 0; i < truePosArr.length; i++) {
            truePosArr[i] /= maxTrue;
            falsePosArr[i] /= maxFalse;
        }

        return new ROC(falsePosArr,truePosArr,thresholdsArr);
    }

    private static TPFP generateTPFPs(boolean[] yPos, double[] yScore) {
        if (yPos.length != yScore.length) {
            throw new IllegalArgumentException("yPos and yScore must be the same length, yPos.length = " + yPos.length + ", yScore.length = " + yScore.length);
        }
        // First sort the predictions by their score
        // and apply that sort to the true labels and the predictions.
        int[] sortedIndices = SortUtil.argsort(yScore,false);
        double[] sortedScore = new double[yScore.length];
        boolean[] sortedPos = new boolean[yPos.length];
        int totalPos = 0;
        for (int i = 0; i < yScore.length; i++) {
            sortedScore[i] = yScore[sortedIndices[i]];
            sortedPos[i] = yPos[sortedIndices[i]];
            if (sortedPos[i]) {
                totalPos++;
            }
        }

        // Find all the differences in the score values as values with
        // the same score need to be compressed into a single ROC point.
        int[] differentIndices = Util.differencesIndices(sortedScore);
        int[] truePosSum = Util.cumulativeSum(sortedPos);

        // Calculate the number of true positives and false positives for each score threshold.
        ArrayList<Integer> truePos = new ArrayList<>();
        ArrayList<Integer> falsePos = new ArrayList<>();
        ArrayList<Double> thresholds = new ArrayList<>();
        for (int i = 0; i < differentIndices.length; i++) {
            thresholds.add(sortedScore[differentIndices[i]]);
            truePos.add(truePosSum[differentIndices[i]]);
            falsePos.add(1+(differentIndices[i] - truePosSum[differentIndices[i]]));
        }

        return new TPFP(falsePos,truePos,thresholds,totalPos);
    }

    private static class TPFP {
        public final List<Integer> falsePos;
        public final List<Integer> truePos;
        public final List<Double> thresholds;
        public final int totalPos;

        public TPFP(List<Integer> falsePos, List<Integer> truePos, List<Double> thresholds, int totalPos) {
            this.falsePos = falsePos;
            this.truePos = truePos;
            this.thresholds = thresholds;
            this.totalPos = totalPos;
        }
    }

    /**
     * Stores the ROC curve as three arrays: the false positive rate, the true positive rate,
     * and the thresholds associated with those rates.
     * <p>
     * By definition if both tpr and fpr are zero for the first value, the threshold is positive infinity.
     * <p>
     * Not yet a record, but it will be one day.
     */
    public static class ROC {
        /**
         * The false positive rate at the corresponding threshold.
         */
        public final double[] fpr;
        /**
         * The true positive rate at the corresponding threshold.
         */
        public final double[] tpr;
        /**
         * The threshold values.
         */
        public final double[] thresholds;

        /**
         * Constructs an ROC curve.
         * @param fpr The false positive rates.
         * @param tpr The true positive rates.
         * @param thresholds The classification thresholds for the relevant rates.
         */
        public ROC(double[] fpr, double[] tpr, double[] thresholds) {
            this.fpr = fpr;
            this.tpr = tpr;
            this.thresholds = thresholds;
        }
    }

    /**
     * Stores the Precision-Recall curve as three arrays: the precisions, the recalls,
     * and the thresholds associated with those values.
     * <p>
     * Not yet a record, but it will be one day.
     */
    public static class PRCurve {
        /**
         * The precision at the corresponding threshold.
         */
        public final double[] precision;
        /**
         * The recall at the corresponding threshold.
         */
        public final double[] recall;
        /**
         * The threshold values.
         */
        public final double[] thresholds;

        /**
         * Constructs a precision-recall curve.
         * @param precision The precisions.
         * @param recall The recalls.
         * @param thresholds The classification thresholds for the precisions and recalls.
         */
        public PRCurve(double[] precision, double[] recall, double[] thresholds) {
            this.precision = precision;
            this.recall = recall;
            this.thresholds = thresholds;
        }
    }
}

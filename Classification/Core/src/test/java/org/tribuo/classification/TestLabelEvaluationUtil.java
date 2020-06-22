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

package org.tribuo.classification;

import org.tribuo.classification.evaluation.LabelEvaluationUtil.PRCurve;
import org.tribuo.classification.evaluation.LabelEvaluationUtil.ROC;
import org.tribuo.util.Util;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.tribuo.classification.evaluation.LabelEvaluationUtil.averagedPrecision;
import static org.tribuo.classification.evaluation.LabelEvaluationUtil.generatePRCurve;
import static org.tribuo.classification.evaluation.LabelEvaluationUtil.generateROCCurve;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestLabelEvaluationUtil {
    private static final double DELTA = 1e-12;

    @Test
    public void testPRC() {
        boolean[] yTrue = new boolean[]{true, false, false, true};
        double[] yScore = new double[]{1, 2, 3, 4};
        PRCurve prc = generatePRCurve(yTrue,yScore);
        assertArrayEquals(new double[]{0.5, 1.0/3.0, 0.5, 1., 1.},prc.precision,DELTA);
        assertArrayEquals(new double[]{1., 0.5, 0.5, 0.5, 0.},prc.recall,DELTA);
        assertArrayEquals(new double[]{1, 2, 3, 4},prc.thresholds,DELTA);
        assertEquals(prc.precision.length, prc.recall.length);
    }

    @Test
    public void testPRCConstantData() {
        boolean[] yTrue = new boolean[100];
        double[] yScore = new double[100];
        Arrays.fill(yTrue,75,100,true);
        Arrays.fill(yScore,1.0);
        assertEquals(0.25,averagedPrecision(yTrue,yScore),DELTA);
    }

    @Test
    public void testPRCToyData() {
        boolean[] yTrue = new boolean[]{false, true};
        double[] yScore = new double[]{0, 1};
        PRCurve prc = generatePRCurve(yTrue, yScore);
        double aucprc = averagedPrecision(yTrue, yScore);
        assertArrayEquals(new double[]{1,1}, prc.precision, DELTA);
        assertArrayEquals(new double[]{1,0}, prc.recall, DELTA);
        assertEquals(1.,aucprc,DELTA);

        yTrue = new boolean[]{false, true};
        yScore = new double[]{1, 0};
        prc = generatePRCurve(yTrue, yScore);
        aucprc = averagedPrecision(yTrue, yScore);
        assertArrayEquals(new double[]{0.5, 0., 1.}, prc.precision, DELTA);
        assertArrayEquals(new double[]{1., 0., 0.}, prc.recall, DELTA);
        assertEquals(0.5,aucprc, DELTA);

        yTrue = new boolean[]{true,false};
        yScore = new double[]{1, 1};
        prc = generatePRCurve(yTrue, yScore);
        aucprc = averagedPrecision(yTrue, yScore);
        assertArrayEquals(new double[]{0.5,1.0}, prc.precision,DELTA);
        assertArrayEquals(new double[]{1.0,0.0}, prc.recall,DELTA);
        assertEquals(aucprc, .5,DELTA);

        yTrue = new boolean[]{true,false};
        yScore = new double[]{1, 0};
        prc = generatePRCurve(yTrue, yScore);
        aucprc = averagedPrecision(yTrue, yScore);
        assertArrayEquals(new double[]{1.0,1.0}, prc.precision,DELTA);
        assertArrayEquals(new double[]{1.0,0.0}, prc.recall,DELTA);
        assertEquals(aucprc, 1.,DELTA);

        yTrue = new boolean[]{true,false};
        yScore = new double[]{0.5,0.5};
        prc = generatePRCurve(yTrue, yScore);
        aucprc = averagedPrecision(yTrue, yScore);
        assertArrayEquals(new double[]{0.5,1.0}, prc.precision,DELTA);
        assertArrayEquals(new double[]{1.0,0.0}, prc.recall,DELTA);
        assertEquals(aucprc, .5,DELTA);

        yTrue = new boolean[]{true,true};
        yScore = new double[]{0.25,0.75};
        prc = generatePRCurve(yTrue, yScore);
        aucprc = averagedPrecision(yTrue,yScore);
        assertArrayEquals(new double[]{1.0,1.0,1.0}, prc.precision, DELTA);
        assertArrayEquals(new double[]{ 1.0, 0.5, 0.0}, prc.recall, DELTA);
        assertEquals(1.,aucprc,DELTA);
    }

    @Test
    public void testROCIncreasing() {
        boolean[] yTrue = new boolean[]{false, false, true, true, true};
        double[] yScore = new double[]{0.1, 0.7, 0.3, 0.4, 0.5};
        ROC roc = generateROCCurve(yTrue, yScore);
        for (int i = 1; i < roc.fpr.length; i++) {
            double fprDiff = roc.fpr[i] - roc.fpr[i-1];
            assertTrue(fprDiff >= 0.0);
            double tprDiff = roc.tpr[i] - roc.tpr[i-1];
            assertTrue(tprDiff >= 0.0);
        }
    }

    @Test
    public void testAUCToyData() {
        boolean[] yTrue;
        double[] yScore;
        ROC roc;
        double auc;

        yTrue = new boolean[]{false, true};
        yScore = new double[]{0, 1};
        roc = generateROCCurve(yTrue, yScore);
        auc = Util.auc(roc.fpr, roc.tpr);
        assertArrayEquals(new double[]{0, 0, 1}, roc.fpr, DELTA);
        assertArrayEquals(new double[]{0, 1, 1}, roc.tpr, DELTA);
        assertEquals(1.0, auc, DELTA);

        yTrue = new boolean[]{false, true};
        yScore = new double[]{1, 0};
        roc = generateROCCurve(yTrue, yScore);
        auc = Util.auc(roc.fpr, roc.tpr);
        assertArrayEquals(new double[]{0, 1, 1}, roc.fpr, DELTA);
        assertArrayEquals(new double[]{0, 0, 1}, roc.tpr, DELTA);
        assertEquals(0.0, auc, DELTA);

        yTrue = new boolean[]{true, false};
        yScore = new double[]{1, 1};
        roc = generateROCCurve(yTrue, yScore);
        auc = Util.auc(roc.fpr, roc.tpr);
        assertArrayEquals(new double[]{0, 1}, roc.fpr, DELTA);
        assertArrayEquals(new double[]{0, 1}, roc.tpr, DELTA);
        assertEquals(0.5, auc, DELTA);

        yTrue = new boolean[]{true, false};
        yScore = new double[]{1, 0};
        roc = generateROCCurve(yTrue, yScore);
        auc = Util.auc(roc.fpr, roc.tpr);
        assertArrayEquals(new double[]{0, 0, 1}, roc.fpr, DELTA);
        assertArrayEquals(new double[]{0, 1, 1}, roc.tpr, DELTA);
        assertEquals(1.0, auc, DELTA);

        yTrue = new boolean[]{true, false};
        yScore = new double[]{0.5, 0.5};
        roc = generateROCCurve(yTrue, yScore);
        auc = Util.auc(roc.fpr, roc.tpr);
        assertArrayEquals(new double[]{0, 1}, roc.fpr, DELTA);
        assertArrayEquals(new double[]{0, 1}, roc.tpr, DELTA);
        assertEquals(0.5, auc, DELTA);
    }

    @Test
    public void testROCExample() {
        boolean[] yTrue = new boolean[]{false, false, true, true};
        double[] yScore = new double[]{0.1, 0.4, 0.35, 0.8};
        ROC roc = generateROCCurve(yTrue, yScore);
        double auc = Util.auc(roc.fpr,roc.tpr);
        assertArrayEquals(new double[]{0.0, 0.5, 0.5, 1.0, 1.0}, roc.tpr, DELTA);
        assertArrayEquals(new double[]{0.0, 0.0, 0.5, 0.5, 1.0}, roc.fpr, DELTA);
        assertArrayEquals(new double[]{Double.POSITIVE_INFINITY, 0.8, 0.4, 0.35, 0.1}, roc.thresholds, DELTA);
        assertEquals(0.75,auc,DELTA);
    }

}

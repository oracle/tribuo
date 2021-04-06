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

package org.tribuo.multilabel;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.evaluation.ClassifierEvaluation;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.multilabel.baseline.IndependentMultiLabelTrainer;
import org.tribuo.multilabel.evaluation.MultiLabelEvaluator;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.tribuo.test.Helpers;
import com.oracle.labs.mlrg.olcut.util.Pair;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 *
 */
public class IndependentMultiLabelTest {

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(LinearSGDTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testIndependentBinaryPredictions() {
        MultiLabelFactory factory = new MultiLabelFactory();
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        IndependentMultiLabelTrainer trainer = new IndependentMultiLabelTrainer(new LogisticRegressionTrainer());
        Model<MultiLabel> model = trainer.train(train);

        List<Prediction<MultiLabel>> predictions = model.predict(test);
        Prediction<MultiLabel> first = predictions.get(0);
        MultiLabel trueLabel = factory.generateOutput("MONKEY,PUZZLE,TREE");
        assertEquals(trueLabel, first.getOutput(), "Predicted labels not equal");
        Map<String, List<Pair<String,Double>>> features = model.getTopFeatures(2);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());

        Helpers.testModelSerialization(model,MultiLabel.class);
    }

    @Test
    public void testMultiLabelConfusionMatrixToStrings() {
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        IndependentMultiLabelTrainer trainer = new IndependentMultiLabelTrainer(
            new LogisticRegressionTrainer());
        Model<MultiLabel> model = trainer.train(train);

        ClassifierEvaluation<MultiLabel> evaluation = new MultiLabelEvaluator()
            .evaluate(model, test);

        System.out.println(evaluation);

        // MultiLabelConfusionMatrix toString() hard to interpret
        final ConfusionMatrix<MultiLabel> mcm = evaluation.getConfusionMatrix();

        System.out.println("original");
        System.out.println(mcm);

        System.out.println("\npretty");
        System.out.println(prettyToString(mcm));

        System.out.println("\nlabelConfusionMatrixToString");
        System.out.println(labelConfusionMatrixToString(mcm));
    }

    public static String prettyToString(ConfusionMatrix<MultiLabel> mcmObject) {
        return mcmObject.getDomain().getDomain().stream()
            .map(multiLabel -> {
                    final int tp = (int) mcmObject.tp(multiLabel);
                    final int fn = (int) mcmObject.fn(multiLabel);
                    final int fp = (int) mcmObject.fp(multiLabel);
                    final int tn = (int) mcmObject.tn(multiLabel);
                    return multiLabel + "\n"
                        + String.format("[tn: %,d fn: %,d]\n", tn, fn)
                        + String.format("[fp: %,d tp: %,d]", fp, tp);
                }
            ).collect(Collectors.joining("\n"));
    }

    public static String labelConfusionMatrixToString(ConfusionMatrix<MultiLabel> mcmObject) {
        ImmutableOutputInfo<MultiLabel> domain = mcmObject.getDomain();
        List<MultiLabel> labelOrder = new ArrayList<>(domain.getDomain());

        StringBuilder sb = new StringBuilder();

        int maxLen = Integer.MIN_VALUE;
        for (MultiLabel multiLabel : labelOrder) {
            maxLen = Math.max(multiLabel.getLabelString().length(), maxLen);
            maxLen = Math.max(String.format(" %,d", (int) mcmObject.support(multiLabel)).length(), maxLen);
        }

        String trueLabelFormat = String.format("%%-%ds", maxLen + 2);
        String predictedLabelFormat = String.format("%%%ds", maxLen + 2);
        String countFormat = String.format("%%,%dd", maxLen + 2);

        //
        // Empty spot in first row for labels on subsequent rows.
        sb.append(String.format(trueLabelFormat, ""));

        //
        // Labels across the top for predicted.
        for (MultiLabel multiLabel : labelOrder) {
            sb.append(String.format(predictedLabelFormat, multiLabel.getLabelString()));
        }
        sb.append('\n');

        for (MultiLabel trueLabel : labelOrder) {
            sb.append(String.format(trueLabelFormat, trueLabel.getLabelString()));
            for (MultiLabel predictedLabel : labelOrder) {
                int confusion = (int) mcmObject.confusion(predictedLabel, trueLabel);
                sb.append(String.format(countFormat, confusion));
            }
            sb.append('\n');
        }

        return sb.toString();
    }
}

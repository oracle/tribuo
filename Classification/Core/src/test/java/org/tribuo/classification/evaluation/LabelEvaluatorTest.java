/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.evaluation.EvaluationRenderer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.tribuo.classification.Utils.label;
import static org.tribuo.classification.Utils.mkPrediction;
import static org.tribuo.classification.evaluation.TestUtils.mkDataset;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class LabelEvaluatorTest {

    @Test
    public void testLabelEvaluator() {
        List<Prediction<Label>> predictions = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c")
        );
        Dataset<Label> dataset = mkDataset(predictions);
        Model<Label> model = DummyClassifierTrainer.createMostFrequentTrainer().train(dataset); // just a noop model

        LabelEvaluation evaluation = new LabelEvaluator()
                .evaluate(model, predictions, dataset.getProvenance());

        Label a = label("a");
        Label b = label("b");
        Label c = label("c");

        assertThrows(UnsupportedOperationException.class, () -> evaluation.AUCROC(a));

        assertEquals(1d, evaluation.precision(a));
        assertEquals(1d, evaluation.recall(a));
        assertEquals(1d, evaluation.f1(a));

        assertEquals(0.5, evaluation.precision(b));
        assertEquals(0.5, evaluation.recall(b));
        assertEquals(0.5, evaluation.f1(b));

        assertEquals(0d, evaluation.precision(c));
        assertEquals(0d, evaluation.recall(c));
        assertEquals(0d, evaluation.f1(c));

        assertEquals(0.5, evaluation.accuracy());
        assertEquals(0.5, evaluation.balancedErrorRate());
        assertEquals(0.5, evaluation.macroAveragedPrecision());
        assertEquals(0.5, evaluation.macroAveragedRecall());
        assertEquals(0.5, evaluation.macroAveragedF1());
        assertEquals(0.5, evaluation.microAveragedPrecision());
        assertEquals(0.5, evaluation.microAveragedRecall());
        assertEquals(0.5, evaluation.microAveragedF1());
    }

    @Test
    public void testEmptyPredictions() {
        Logger l = Logger.getLogger(ConfusionMetrics.class.getName());
        l.setLevel(Level.SEVERE);
        List<Prediction<Label>> predictions = new ArrayList<>();
        Dataset<Label> dataset = mkDataset(predictions); // just an empty dataset
        Model<Label> model = DummyClassifierTrainer.createMostFrequentTrainer().train(dataset); // just a noop model
        LabelEvaluation evaluation = new LabelEvaluator()
                .evaluate(model, predictions, dataset.getProvenance());
        assertEquals(Double.NaN, evaluation.tp());
        assertEquals(Double.NaN, evaluation.fp());
        assertEquals(Double.NaN, evaluation.tn());
        assertEquals(Double.NaN, evaluation.fn());
        assertEquals(Double.NaN, evaluation.accuracy());
        assertEquals(Double.NaN, evaluation.balancedErrorRate());
        assertThrows(IllegalArgumentException.class, () -> evaluation.precision(label("a")));
        assertEquals(Double.NaN, evaluation.macroAveragedPrecision());
        assertEquals(Double.NaN, evaluation.microAveragedPrecision());
        assertEquals(Double.NaN, evaluation.macroAveragedRecall());
        assertEquals(Double.NaN, evaluation.macroAveragedF1());
        assertEquals(Double.NaN, evaluation.microAveragedRecall());
        assertEquals(Double.NaN, evaluation.microAveragedF1());
    }

    @Test
    public void testDefaultToString() {
        List<Prediction<Label>> trPreds = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c"),
                mkPrediction("d", "a")
        );
        Dataset<Label> dataset = mkDataset(trPreds);
        Model<Label> model = DummyClassifierTrainer.createMostFrequentTrainer().train(dataset);

        List<Prediction<Label>> testPreds = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c"),
                mkPrediction("d", "d")
        );

        LabelEvaluation evaluation = new LabelEvaluator()
                .evaluate(model, testPreds, dataset.getProvenance());

        List<Label> lblOrder = new ArrayList<>();
        lblOrder.add(new Label("a"));
        lblOrder.add(new Label("b"));
        lblOrder.add(new Label("c"));
        lblOrder.add(new Label("d"));

        evaluation.getConfusionMatrix().setLabelOrder(lblOrder);

        // Uses String.format to respect JVM locale
        String expected =     "Class                           n          tp          fn          fp      recall        prec          f1\n" +
                String.format("a                               1           1           0           0       %1.3f       %1.3f       %1.3f\n",1.0,1.0,1.0) +
                String.format("b                               2           1           1           1       %1.3f       %1.3f       %1.3f\n",0.5,0.5,0.5) +
                String.format("c                               1           0           1           1       %1.3f       %1.3f       %1.3f\n",0.0,0.0,0.0) +
                String.format("d                               1           1           0           0       %1.3f       %1.3f       %1.3f\n",1.0,1.0,1.0) +
                String.format("Total                           5           3           2           2\n") +
                String.format("Accuracy                                                                    %1.3f\n",0.6) +
                String.format("Micro Average                                                               %1.3f       %1.3f       %1.3f\n",0.6,0.6,0.6) +
                String.format("Macro Average                                                               %1.3f       %1.3f       %1.3f\n",0.625,0.625,0.625) +
                String.format("Balanced Error Rate                                                         %1.3f",0.375);

        String actual = evaluation.toString();
        actual = actual.replaceAll("\\r+", "");
        assertEquals(expected, actual);

        String reducedExpected =     "Class                           n          tp          fn          fp      recall        prec          f1\n" +
            String.format("c                               1           0           1           1       %1.3f       %1.3f       %1.3f\n",0.0,0.0,0.0) +
            String.format("b                               2           1           1           1       %1.3f       %1.3f       %1.3f\n",0.5,0.5,0.5) +
            String.format("Total                           3           1           2           2\n") +
            String.format("Accuracy                                                                    %1.3f\n",0.6) +
            String.format("Micro Average                                                               %1.3f       %1.3f       %1.3f\n",0.6,0.6,0.6) +
            String.format("Macro Average                                                               %1.3f       %1.3f       %1.3f\n",0.625,0.625,0.625) +
            String.format("Balanced Error Rate                                                         %1.3f",0.375);

        lblOrder.clear();
        lblOrder.add(new Label("c"));
        lblOrder.add(new Label("b"));
        evaluation.getConfusionMatrix().setLabelOrder(lblOrder);
        String reducedActual = evaluation.toString();
        reducedActual = reducedActual.replaceAll("\\r+", "");
        assertEquals(reducedExpected, reducedActual);
    }

    @Test
    public void testDefaultToHTML() {
        List<Prediction<Label>> trPreds = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c"),
                mkPrediction("d", "a")
        );
        Dataset<Label> dataset = mkDataset(trPreds);
        Model<Label> model = DummyClassifierTrainer.createMostFrequentTrainer().train(dataset);

        List<Prediction<Label>> testPreds = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c"),
                mkPrediction("d", "d")
        );

        LabelEvaluation evaluation = new LabelEvaluator()
                .evaluate(model, testPreds, dataset.getProvenance());

        List<Label> lblOrder = new ArrayList<>();
        lblOrder.add(new Label("a"));
        lblOrder.add(new Label("b"));
        lblOrder.add(new Label("c"));
        lblOrder.add(new Label("d"));

        evaluation.getConfusionMatrix().setLabelOrder(lblOrder);

        // Uses String.format to respect JVM locale
        String expected = "<table>\n" +
                "<tr>\n" +
                "<th>Class</th><th>n</th> <th>%</th> <th>tp</th> <th>fn</th> <th>fp</th> <th>Recall</th> <th>Precision</th> <th>F1</th>\n" +
                "</tr>\n" +
                String.format("<tr><td><code>a</code></td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">     %1.1f%%</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td>\n</tr>",
                    1,7.7,1,0,0,1.0,1.0,1.0) +
                String.format("<tr><td><code>b</code></td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">    %1.1f%%</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td>\n</tr>",
                    2,15.4,1,1,1,0.5,0.5,0.5) +
                String.format("<tr><td><code>c</code></td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">     %1.1f%%</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td>\n</tr>",
                    1,7.7,0,1,1,0.0,0.0,0.0) +
                String.format("<tr><td><code>d</code></td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">     %1.1f%%</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td>\n</tr>",
                    1,7.7,1,0,0,1.0,1.0,1.0) +
                "<tr><td>Total</td><td style=\"text-align:right\">          13</td><td style=\"text-align:right\"></td><td style=\"text-align:right\">           3</td><td style=\"text-align:right\">           2</td><td style=\"text-align:right\">           2</td>\n" +
                "<td colspan=\"4\"></td></tr>\n" +
                String.format("<tr><td>Accuracy</td><td style=\"text-align:right\" colspan=\"6\">   %1.3f</td>\n",0.6) +
                "<td colspan=\"4\"></td></tr>\n" +
                String.format("<tr><td>Micro Average</td><td style=\"text-align:right\" colspan=\"6\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td><td style=\"text-align:right\">   %1.3f</td>\n",
                        0.6,0.6,0.6) +
                "</tr></table>";
        String actual = evaluation.toHTML();
        actual = actual.replaceAll("\\r+", "");
        assertEquals(expected, actual);
    }

    @Test
    public void testCustom() {
        List<Prediction<Label>> trPreds = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c"),
                mkPrediction("d", "a")
        );
        Dataset<Label> dataset = mkDataset(trPreds);
        Model<Label> model = DummyClassifierTrainer.createMostFrequentTrainer().train(dataset);

        List<Prediction<Label>> testPreds = Arrays.asList(
                mkPrediction("a", "a"),
                mkPrediction("c", "b"),
                mkPrediction("b", "b"),
                mkPrediction("b", "c"),
                mkPrediction("d", "d")
        );

        LabelEvaluation evaluation = new LabelEvaluator()
                .evaluate(model, testPreds, dataset.getProvenance());

        EvaluationRenderer<Label, LabelEvaluation> custom = (LabelEvaluation eval) -> {
            StringBuilder sb = new StringBuilder();
            sb.append("micro avg: ");
            sb.append(String.format("%2.2f %2.2f %2.2f",
                    eval.microAveragedRecall(),
                    eval.microAveragedPrecision(),
                    eval.microAveragedF1()));
            return sb.toString();
        };

        // Use String.format to respect JVM locale
        String expected = String.format("micro avg: %2.2f %2.2f %2.2f", 0.60, 0.60, 0.60);
        assertEquals(expected, custom.apply(evaluation));
    }

}
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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.evaluation.DescriptiveStats;
import org.tribuo.evaluation.EvaluationAggregator;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static java.lang.System.out;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class EvaluationAggregationTests {

    private static final LabelFactory factory = new LabelFactory();

    /**
     * A driver that exercises the classification evaluation system.
     * @param args Ignores arguments.
     */
    public static void main(String[] args) {
        xval();
        summarizeF1AcrossModels();
        summarizeF1AcrossModels_v2();
        bestModel();
        bestDataset();
    }

    public static void xval() {
        Trainer<Label> trainer = DummyClassifierTrainer.createUniformTrainer(1L);

        Pair<Dataset<Label>, Dataset<Label>> datasets = LabelledDataGenerator.denseTrainTest();
        Dataset<Label> trainData = datasets.getA();

        Evaluator<Label, LabelEvaluation> evaluator = factory.getEvaluator();

        CrossValidation<Label, LabelEvaluation> xval = new CrossValidation<>(
                trainer,
                trainData,
                evaluator,
                5
        );
        List<Pair<LabelEvaluation, Model<Label>>> results = xval.evaluate();

        List<LabelEvaluation> evals = results
                .stream()
                .map(Pair::getA)
                .collect(Collectors.toList());

        // Summarize across everything
        Map<MetricID<Label>, DescriptiveStats> summary = EvaluationAggregator.summarize(evals);
        List<MetricID<Label>> keys = new ArrayList<>(summary.keySet())
                .stream()
                .sorted(Comparator.comparing(Pair::getB))
                .collect(Collectors.toList());
        for (MetricID<Label> key : keys) {
            DescriptiveStats stats = summary.get(key);
            out.printf("%-10s  %.5f (%.5f)%n", key, stats.getMean(), stats.getStandardDeviation());
        }

        // Summarize across macro F1s only
        DescriptiveStats macroF1Summary = EvaluationAggregator.summarize(evals, LabelEvaluation::macroAveragedF1);
        out.println(macroF1Summary);

        Pair<Integer, Double> argmax = EvaluationAggregator.argmax(evals, LabelEvaluation::macroAveragedF1);
        Model<Label> bestF1 = results.get(argmax.getA()).getB();

        LabelEvaluation testEval = evaluator.evaluate(bestF1, datasets.getB());
        System.out.println(testEval);
    }

    public static void summarizeF1AcrossModels() {
        Pair<Dataset<Label>, Dataset<Label>> pair = LabelledDataGenerator.denseTrainTest();

        List<Model<Label>> models = Arrays.asList(
                DummyClassifierTrainer.createUniformTrainer(1L).train(pair.getA()),
                DummyClassifierTrainer.createUniformTrainer(2L).train(pair.getA()),
                DummyClassifierTrainer.createUniformTrainer(3L).train(pair.getA())
        );

//        // summary for label 'Foo'
//        MetricTarget<Label> target = new MetricTarget<Label>(factory.generateOutput("Foo"));
//        LabelMetric metric = LabelMetrics.Default.F1.forTarget(target);
//        DescriptiveStats summary = EvaluationAggregator.summarize(metric, models, pair.getB());
//        out.println("\n\nLabel 'Foo' across models:\n" + summary.toString());

        //
        // summary for macro F1
        LabelMetric macroF1 = LabelMetrics
                .F1
                .forTarget(MetricTarget.macroAverageTarget());

        DescriptiveStats summary = EvaluationAggregator.summarize(macroF1, models, pair.getB());
        out.println("\nMacro F1 across models:\n" + summary.toString());
    }

    public static void summarizeF1AcrossModels_v2() {
        Pair<Dataset<Label>, Dataset<Label>> pair = LabelledDataGenerator.denseTrainTest();
        List<Model<Label>> models = Arrays.asList(
                DummyClassifierTrainer.createUniformTrainer(1L).train(pair.getA()),
                DummyClassifierTrainer.createUniformTrainer(2L).train(pair.getA()),
                DummyClassifierTrainer.createUniformTrainer(3L).train(pair.getA())
        );

        Evaluator<Label, LabelEvaluation> evaluator = factory.getEvaluator();

        Map<MetricID<Label>, DescriptiveStats> summaries = EvaluationAggregator.summarize(evaluator, models, pair.getB());

        MetricID<Label> macroF1 = LabelMetrics
                .F1
                .forTarget(MetricTarget.macroAverageTarget())
                .getID();

        DescriptiveStats summary = summaries.get(macroF1);
        out.println("\nMacro F1 across models (V2):\n" + summary.toString());
    }

    public static Model<Label> bestModel() {
        Pair<Dataset<Label>, Dataset<Label>> pair = LabelledDataGenerator.denseTrainTest();
        List<Model<Label>> models = Arrays.asList(
                DummyClassifierTrainer.createUniformTrainer(1L).train(pair.getA()),
                DummyClassifierTrainer.createUniformTrainer(2L).train(pair.getA()),
                DummyClassifierTrainer.createUniformTrainer(3L).train(pair.getA())
        );

        Evaluator<Label, LabelEvaluation> evaluator = factory.getEvaluator();
        List<LabelEvaluation> evals = models
                .stream()
                .map(model -> evaluator.evaluate(model, pair.getB()))
                .collect(Collectors.toList());

        Pair<Integer, Double> ivmax = EvaluationAggregator.argmax(evals, LabelEvaluation::macroAveragedF1);
        out.printf("%nBest Macro F1 @ model #%d = %f%n", ivmax.getA(), ivmax.getB());

        return models.get(ivmax.getA());
    }

    public static Dataset<Label> bestDataset() {
        Pair<Dataset<Label>, Dataset<Label>> pair = LabelledDataGenerator.denseTrainTest(-0.3);
        Model<Label> model = DummyClassifierTrainer.createUniformTrainer(1L).train(pair.getA());

        List<Dataset<Label>> datasets = Arrays.asList(
                LabelledDataGenerator.denseTrainTest(-1.0).getB(),
                LabelledDataGenerator.denseTrainTest(-0.5).getB(),
                LabelledDataGenerator.denseTrainTest(-0.1).getB()
        );

        Evaluator<Label, LabelEvaluation> evaluator = factory.getEvaluator();

        List<LabelEvaluation> evals = datasets
                .stream()
                .map(data -> evaluator.evaluate(model, data))
                .collect(Collectors.toList());

        Pair<Integer, Double> ivmax = EvaluationAggregator.argmax(evals, LabelEvaluation::macroAveragedF1);

        out.printf("%nBest Macro F1 @ dataset #%d = %f%n", ivmax.getA(), ivmax.getB());
        return datasets.get(ivmax.getA());
    }

    @Test
    public void summarizeF1AcrossDatasets() {
        Pair<Dataset<Label>, Dataset<Label>> pair = LabelledDataGenerator.denseTrainTest(-0.3);
        Model<Label> model = DummyClassifierTrainer.createMostFrequentTrainer().train(pair.getA());

        List<Dataset<Label>> datasets = Arrays.asList(
                LabelledDataGenerator.denseTrainTest(-1.0).getB(),
                LabelledDataGenerator.denseTrainTest(-0.5).getB(),
                LabelledDataGenerator.denseTrainTest(-0.1).getB()
        );

        LabelMetric macroF1Metric = LabelMetrics
                .F1
                .forTarget(MetricTarget.macroAverageTarget());

        DescriptiveStats summary = EvaluationAggregator.summarize(macroF1Metric, model, datasets);

        List<Double> macroF1 = new ArrayList<>();
        for (Dataset<Label> d : datasets) {
            LabelEvaluation eval = factory.getEvaluator().evaluate(model, datasets.get(0));
            macroF1.add(eval.macroAveragedF1());
        }

        DescriptiveStats otherSummary = new DescriptiveStats(macroF1);
        assertEquals(summary, otherSummary);
    }

    @Test
    public void summarizeF1AcrossDatasets_v2() {
        Pair<Dataset<Label>, Dataset<Label>> pair = LabelledDataGenerator.denseTrainTest(-0.3);
        Model<Label> model = DummyClassifierTrainer.createMostFrequentTrainer().train(pair.getA());

        List<Dataset<Label>> datasets = Arrays.asList(
                LabelledDataGenerator.denseTrainTest(-1.0).getB(),
                LabelledDataGenerator.denseTrainTest(-0.5).getB(),
                LabelledDataGenerator.denseTrainTest(-0.1).getB()
        );

        Evaluator<Label, LabelEvaluation> evaluator = factory.getEvaluator();

        Map<MetricID<Label>, DescriptiveStats> summaries = EvaluationAggregator.summarize(evaluator, model, datasets);

        MetricID<Label> macroF1 = LabelMetrics
                .F1
                .forTarget(MetricTarget.macroAverageTarget())
                .getID();

        DescriptiveStats summary = summaries.get(macroF1);

        // Can also do this:
        List<LabelEvaluation> evals = datasets
                .stream()
                .map(dataset -> evaluator.evaluate(model, dataset))
                .collect(Collectors.toList());
        Map<MetricID<Label>, DescriptiveStats> summaries2 = EvaluationAggregator.summarize(evals);

        assertEquals(summaries, summaries2);
    }

    /**
     * Use EvaluationAggregator to summarize model outputs across several metrics (you can compute e.g., macro precision this way)
     */
    @Test
    public void macroPrec() {
        Pair<Dataset<Label>, Dataset<Label>> pair = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = DummyClassifierTrainer.createUniformTrainer(1L).train(pair.getA());
        List<LabelMetric> metrics = Arrays.asList(
                LabelMetrics.PRECISION.forTarget(new MetricTarget<>(factory.generateOutput("Foo"))),
                LabelMetrics.PRECISION.forTarget(new MetricTarget<>(factory.generateOutput("Bar"))),
                LabelMetrics.PRECISION.forTarget(new MetricTarget<>(factory.generateOutput("Baz"))),
                LabelMetrics.PRECISION.forTarget(new MetricTarget<>(factory.generateOutput("Quux")))
        );
        List<Prediction<Label>> predictions = model.predict(pair.getB());
        DescriptiveStats summary = EvaluationAggregator.summarize(metrics, model, predictions);
        double macroPrecision = summary.getMean();
        // Alternatively...
        LabelEvaluation evaluation = factory.getEvaluator().evaluate(model, predictions, pair.getB().getProvenance());
        // they should be the same
        assertEquals(evaluation.macroAveragedPrecision(), macroPrecision);
    }

}
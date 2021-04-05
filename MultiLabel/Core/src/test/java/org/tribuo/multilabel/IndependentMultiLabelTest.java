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

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.ClassifierEvaluation;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.classification.evaluation.ConfusionMetrics;
import org.tribuo.classification.evaluation.LabelConfusionMatrix;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluationUtil;
import org.tribuo.classification.evaluation.LabelMetric;
import org.tribuo.classification.evaluation.LabelMetrics;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.evaluation.metrics.EvaluationMetric.Average;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.evaluation.metrics.MetricTarget;
import org.tribuo.impl.ArrayExample;
import org.tribuo.multilabel.baseline.IndependentMultiLabelTrainer;
import org.tribuo.multilabel.evaluation.MultiLabelEvaluator;
import org.tribuo.multilabel.example.MultiLabelDataGenerator;
import org.tribuo.provenance.EvaluationProvenance;
import org.tribuo.test.Helpers;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
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

    // MultiLabelConfusionMatrix toString() is hard to interpret - convert a MultiLabel evaluation
    // to a Label evaluation
    @Test
    public void multiLabelAsLabel() {
        Dataset<MultiLabel> train = MultiLabelDataGenerator.generateTrainData();
        Dataset<MultiLabel> test = MultiLabelDataGenerator.generateTestData();

        IndependentMultiLabelTrainer trainer = new IndependentMultiLabelTrainer(
            new LogisticRegressionTrainer());
        Model<MultiLabel> model = trainer.train(train);

        ClassifierEvaluation<MultiLabel> evaluation = new MultiLabelEvaluator()
            .evaluate(model, test);

        System.out.println(evaluation);
        // MultiLabelConfusionMatrix toString() hard to interpret
        System.out.println(evaluation.getConfusionMatrix());

        // given MultiLabelModel model
        final List<Prediction<Label>> predictions = asLabelPredictions(evaluation.getPredictions());

        final LabelConfusionMatrix labelConfusionMatrix = labelConfusionMatrix(model, test);

        // model
        final Set<LabelMetric> labelMetrics = createMetrics(model);

        final Map<MetricID<Label>, Double> results = computeMetrics(
            labelConfusionMatrix, labelMetrics, predictions);

        final LabelEvaluation labelEvaluation = labelEvaluation(predictions, results,
            labelConfusionMatrix,
            model.generatesProbabilities());
        System.out.println(labelEvaluation);
        System.out.println(labelEvaluation.getConfusionMatrix());
    }

    private Map<MetricID<Label>, Double> computeMetrics(
        final LabelConfusionMatrix labelConfusionMatrix, final Set<LabelMetric> labelMetrics,
        final List<Prediction<Label>> predictions) {
        return Collections.unmodifiableMap(
            labelMetrics.stream().collect(
                Collectors.toMap(
                    labelMetric -> new MetricID<>(labelMetric.getTarget(), labelMetric.getName()),
                    labelMetric -> {
                        final LabelMetrics aLabelMetrics = LabelMetrics
                            .valueOf(labelMetric.getName());
                        final MetricTarget<Label> tgt = labelMetric.getTarget();
                        switch (aLabelMetrics) {
                        case TP:
                            return ConfusionMetrics.tp(tgt, labelConfusionMatrix);
                        case FP:
                            return ConfusionMetrics.fp(tgt, labelConfusionMatrix);
                        case TN:
                            return ConfusionMetrics.tn(tgt, labelConfusionMatrix);
                        case FN:
                            return ConfusionMetrics.fn(tgt, labelConfusionMatrix);
                        case PRECISION:
                            return ConfusionMetrics.precision(tgt, labelConfusionMatrix);
                        case RECALL:
                            return ConfusionMetrics.recall(tgt, labelConfusionMatrix);
                        case F1:
                            return ConfusionMetrics.f1(tgt, labelConfusionMatrix);
                        case ACCURACY:
                            return ConfusionMetrics.accuracy(tgt, labelConfusionMatrix);
                        case BALANCED_ERROR_RATE:
                            return ConfusionMetrics.balancedErrorRate(labelConfusionMatrix);
//                      case AUCROC:
//                          return LabelMetrics.AUCROC(tgt, predictions);
//                      case AVERAGED_PRECISION:
//                          return LabelMetrics.averagedPrecision(tgt, predictions);
                      default:
                          return Double.NaN;
                          }
                    })));
    }

    // create a LabelEvaluation (mostly copy-paste from LabelEvaluationImpl)
    private static LabelEvaluation labelEvaluation(
        final List<Prediction<Label>> predictions,
        final Map<MetricID<Label>, Double> results,
        final LabelConfusionMatrix labelConfusionMatrix,
        final boolean modelGeneratesProbabilities) {
        return new LabelEvaluation() {
            @Override
            public EvaluationProvenance getProvenance() {
                throw new UnsupportedOperationException();
            }

            @Override
            public List<Prediction<Label>> getPredictions() {
                return predictions;
            }

            @Override
            public Map<MetricID<Label>, Double> asMap() {
                return Collections.unmodifiableMap(results);
            }

            @Override
            public double averagedPrecision(Label label) {
                if (!modelGeneratesProbabilities) {
                    throw new UnsupportedOperationException(
                        "averaged precision score not available for models that do not generate probabilities");
                }
                return get(label, LabelMetrics.AVERAGED_PRECISION);
            }

            /**
             * Note: precision recall curve is not stored in the underlying map, so it won't show up in aggregation.
             * @param label The label to use.
             * @return A PRCurve for that label.
             */
            @Override
            public LabelEvaluationUtil.PRCurve precisionRecallCurve(Label label) {
                return LabelMetrics.precisionRecallCurve(label, getPredictions());
            }

            @Override
            public double AUCROC(Label label) {
                if (!modelGeneratesProbabilities) {
                    throw new UnsupportedOperationException(
                        "AUCROC score not available for models that do not generate probabilities");
                }
                return get(label, LabelMetrics.AUCROC);
            }

            /**
             * Note: averageAUCROC is not stored in the underlying map, so it won't show up in aggregation.
             * @param weighted If true weight by the class counts, if false use a macro average.
             * @return The average AUCROC.
             */
            @Override
            public double averageAUCROC(boolean weighted) {
                if (!modelGeneratesProbabilities) {
                    throw new UnsupportedOperationException(
                        "AUCROC score not available for models that do not generate probabilities");
                }
                double sum = 0.0;
                double weightSum = 0.0;
                for (Label l : getConfusionMatrix().getDomain().getDomain()) {
                    double currentValue = get(l, LabelMetrics.AUCROC);
                    double currentWeight = weighted ? getConfusionMatrix().support(l) : 1.0;
                    sum += currentWeight * currentValue;
                    weightSum += currentWeight;
                }
                return sum / weightSum;
            }

            /**
             * Note: confusion is not stored in the underlying map, so it won't show up in aggregation.
             * @param predictedLabel The predicted label.
             * @param trueLabel The true label.
             * @return The number of times that {@code predictedLabel} was predicted for <code>trueLabel</code>.
             */
            @Override
            public double confusion(Label predictedLabel, Label trueLabel) {
                return getConfusionMatrix().confusion(predictedLabel, trueLabel);
            }

            @Override
            public double tp(Label label) {
                return get(label, LabelMetrics.TP);
            }

            @Override
            public double tp() {
                return get(Average.MICRO, LabelMetrics.TP);
            }

            @Override
            public double macroTP() {
                return get(Average.MACRO, LabelMetrics.TP);
            }

            @Override
            public double fp(Label label) {
                return get(label, LabelMetrics.FP);
            }

            @Override
            public double fp() {
                return get(Average.MICRO, LabelMetrics.FP);
            }

            @Override
            public double macroFP() {
                return get(Average.MACRO, LabelMetrics.FP);
            }

            @Override
            public double tn(Label label) {
                return get(label, LabelMetrics.TN);
            }

            @Override
            public double tn() {
                return get(Average.MICRO, LabelMetrics.TN);
            }

            @Override
            public double macroTN() {
                return get(Average.MACRO, LabelMetrics.TN);
            }

            @Override
            public double fn(Label label) {
                return get(label, LabelMetrics.FN);
            }

            @Override
            public double fn() {
                return get(Average.MICRO, LabelMetrics.FN);
            }

            @Override
            public double macroFN() {
                return get(Average.MACRO, LabelMetrics.FN);
            }

            @Override
            public double precision(Label label) {
                return get(label, LabelMetrics.PRECISION);
            }

            @Override
            public double microAveragedPrecision() {
                return get(Average.MICRO, LabelMetrics.PRECISION);
            }

            @Override
            public double macroAveragedPrecision() {
                return get(Average.MACRO, LabelMetrics.PRECISION);
            }

            @Override
            public double recall(Label label) {
                return get(label, LabelMetrics.RECALL);
            }

            @Override
            public double microAveragedRecall() {
                return get(Average.MICRO, LabelMetrics.RECALL);
            }

            @Override
            public double macroAveragedRecall() {
                return get(Average.MACRO, LabelMetrics.RECALL);
            }

            @Override
            public double f1(Label label) {
                return get(label, LabelMetrics.F1);
            }

            @Override
            public double microAveragedF1() {
                return get(Average.MICRO, LabelMetrics.F1);
            }

            @Override
            public double macroAveragedF1() {
                return get(Average.MACRO, LabelMetrics.F1);
            }

            @Override
            public double accuracy() {
                return get(Average.MICRO, LabelMetrics.ACCURACY);
            }

            @Override
            public double accuracy(Label label) {
                return get(label, LabelMetrics.ACCURACY);
            }

            @Override
            public double balancedErrorRate() {
                // Target doesn't matter for balanced error rate, so we just use Average.macro
                // as it's the macro averaged recall.
                return get(Average.MACRO, LabelMetrics.BALANCED_ERROR_RATE);
            }

            @Override
            public ConfusionMatrix<Label> getConfusionMatrix() {
                return labelConfusionMatrix;
            }

            /**
             * This produces a formatted String suitable for a terminal.
             * @return A formatted String representing this {@code LabelEvaluationImpl}.
             */
            @Override
            public String toString() {
                return LabelEvaluation.toFormattedString(this);
            }

            private double get(MetricTarget<Label> tgt, LabelMetrics metric) {
                return get(metric.forTarget(tgt).getID());
            }

            private double get(Label label, LabelMetrics metric) {
                return get(metric
                    .forTarget(new MetricTarget<>(label))
                    .getID());
            }

            private double get(Average avg, LabelMetrics metric) {
                return get(metric
                    .forTarget(new MetricTarget<>(avg))
                    .getID());
            }
        };
    }

    private static LabelConfusionMatrix labelConfusionMatrix(
        final Model<MultiLabel> model,
        final Dataset<MultiLabel> dataset) {
        return new LabelConfusionMatrix(
            asImmutableOutputInfoLabel(model.getOutputIDInfo()),
            asLabelPredictions(model.predict(dataset))
        );
    }

    // part of LabelEvaluator
    private static Set<LabelMetric> createMetrics(final Model<MultiLabel> model) {
        Set<LabelMetric> metrics = new HashSet<>();
        //
        // Populate labelwise values
        for (final MultiLabel multiLabel : model.getOutputIDInfo().getDomain()) {
            for (final Label label : multiLabel.getLabelSet()) {
                MetricTarget<Label> tgt = new MetricTarget<>(label);
                metrics.add(LabelMetrics.TP.forTarget(tgt));
                metrics.add(LabelMetrics.FP.forTarget(tgt));
                metrics.add(LabelMetrics.TN.forTarget(tgt));
                metrics.add(LabelMetrics.FN.forTarget(tgt));
                metrics.add(LabelMetrics.PRECISION.forTarget(tgt));
                metrics.add(LabelMetrics.RECALL.forTarget(tgt));
                metrics.add(LabelMetrics.F1.forTarget(tgt));
                metrics.add(LabelMetrics.ACCURACY.forTarget(tgt));
                if (model.generatesProbabilities()) {
                    metrics.add(LabelMetrics.AUCROC.forTarget(tgt));
                    metrics.add(LabelMetrics.AVERAGED_PRECISION.forTarget(tgt));
                }
            }

            //
            // Populate averaged values.
            MetricTarget<Label> micro = MetricTarget.microAverageTarget();
            metrics.add(LabelMetrics.TP.forTarget(micro));
            metrics.add(LabelMetrics.FP.forTarget(micro));
            metrics.add(LabelMetrics.TN.forTarget(micro));
            metrics.add(LabelMetrics.FN.forTarget(micro));
            metrics.add(LabelMetrics.PRECISION.forTarget(micro));
            metrics.add(LabelMetrics.RECALL.forTarget(micro));
            metrics.add(LabelMetrics.F1.forTarget(micro));
            metrics.add(LabelMetrics.ACCURACY.forTarget(micro));

            MetricTarget<Label> macro = MetricTarget.macroAverageTarget();
            metrics.add(LabelMetrics.TP.forTarget(macro));
            metrics.add(LabelMetrics.FP.forTarget(macro));
            metrics.add(LabelMetrics.TN.forTarget(macro));
            metrics.add(LabelMetrics.FN.forTarget(macro));
            metrics.add(LabelMetrics.PRECISION.forTarget(macro));
            metrics.add(LabelMetrics.RECALL.forTarget(macro));
            metrics.add(LabelMetrics.F1.forTarget(macro));
            metrics.add(LabelMetrics.ACCURACY.forTarget(macro));

            // Target doesn't matter for balanced error rate, so we just use
            // average.macro as it's the macro average of recalls.
            metrics.add(LabelMetrics.BALANCED_ERROR_RATE.forTarget(macro));
        }
        return metrics;
    }

    // other heuristics could be implemented to make this decision
    private static Label pickTrueLabel(final Label predicted, final Set<Label> remainingTrueLabels,
        final Set<Label> trueLabels) {
        if (trueLabels.contains(predicted)) {
            remainingTrueLabels.remove(predicted);
            return predicted;
        } else if (!remainingTrueLabels.isEmpty()) {
            final Label trueLabel = remainingTrueLabels.iterator().next();
            remainingTrueLabels.remove(trueLabel);
            return trueLabel;
        } else {
            if (trueLabels.size() == 1) {
                return trueLabels.iterator().next();
            }
            return LabelFactory.UNKNOWN_LABEL;
        }
    }

    private static List<Prediction<Label>>
    asLabelPredictions(List<Prediction<MultiLabel>> multiLabelPredictions) {
        return Collections.unmodifiableList(multiLabelPredictions.stream()
            .flatMap(prediction -> {
                final Set<Label> predictionLabels = prediction.getOutput().getLabelSet();
                final Map<String, MultiLabel> predictionMultiLabelOutputScores = prediction
                    .getOutputScores();
                final int numUsed = prediction.getNumActiveFeatures();
                final Example<MultiLabel> multiLabelExample = prediction.getExample();
                final Map<String, Object> exampleMetadata = multiLabelExample.getMetadata();
                final Set<Label> mutableTrueLabels = multiLabelExample.getOutput().getLabelSet();
                final Set<Label> trueLabels = Collections
                    .unmodifiableSet(new HashSet<>(mutableTrueLabels));
                final boolean probability = prediction.hasProbabilities();
                return predictionLabels.stream()
                    .map(predictionOutputLabel -> {
                        final Label exampleLabel = pickTrueLabel(predictionOutputLabel,
                            mutableTrueLabels,
                            trueLabels);
                        final Example<Label> labelExample = new ArrayExample<>(
                            exampleLabel,
                            multiLabelExample,
                            multiLabelExample.getWeight()
                        );
                        exampleMetadata.forEach(labelExample::setMetadataValue);
                        assert !probability : "probability assumed false is true";
                        assert predictionMultiLabelOutputScores.isEmpty() :
                            "output scores not yet supported";
                        return new Prediction<>(
                            predictionOutputLabel,
                            numUsed,
                            labelExample
                        );
                    });
            }).collect(Collectors.toList()));
    }

    private static ImmutableOutputInfo<Label>
    asImmutableOutputInfoLabel(ImmutableOutputInfo<MultiLabel> multiLabelOutputInfo) {
        // ImmutableLabelInfo
        Map<Integer, String> idLabelNameMap = new HashMap<>();
        Map<String, Integer> labelNameIDMap = new HashMap<>();

        // LabelInfo fields
        Map<String, MutableLong> labelNameCounts = new HashMap<>();

        Map<String, MultiLabel> multiLabels = Collections.unmodifiableMap(
            multiLabelOutputInfo.getDomain().stream()
                .collect(Collectors.toMap(
                    MultiLabel::getLabelString,
                    Function.identity()
                )));

        Map<String, Label> labels = Collections.unmodifiableMap(
            multiLabelOutputInfo.getDomain().stream()
                .flatMap(multiLabel -> multiLabel.getLabelSet().stream())
                .collect(Collectors.toMap(
                    Label::getLabel,
                    Function.identity()
                )));

        final int unknownCount = multiLabelOutputInfo.getUnknownCount();

        final Iterable<Pair<String, Long>> outputCountsIterable = () ->
            StreamSupport.stream(multiLabelOutputInfo.outputCountsIterable().spliterator(), false)
                .flatMap((Pair<String, Long> multiLabelNameCount) -> {
                    final String multiLabelName = multiLabelNameCount.getA();
                    final long multiLabelCount = multiLabelNameCount.getB();
                    return multiLabels.get(multiLabelName).getLabelSet().stream()
                        .map(label -> new Pair<>(label.getLabel(), multiLabelCount));
                }).iterator();

        final long totalCount = StreamSupport.stream(outputCountsIterable.spliterator(), false)
            .mapToLong(labelNameCount -> {
                final long labelCount = labelNameCount.getB();
                labelNameCounts
                    .computeIfAbsent(labelNameCount.getA(), k -> new MutableLong())
                    .increment(labelCount);
                return labelCount;
            }).sum();

        int counter = 0;
        for (Map.Entry<String, MutableLong> e : labelNameCounts.entrySet()) {
            idLabelNameMap.put(counter, e.getKey());
            labelNameIDMap.put(e.getKey(), counter);
            counter++;
        }

        return new ImmutableOutputInfo<Label>() {
            @Override
            public int getID(final Label output) {
                return labelNameIDMap.get(output.getLabel());
            }

            @Override
            public Label getOutput(final int id) {
                return labels.get(idLabelNameMap.get(id));
            }

            @Override
            public long getTotalObservations() {
                return totalCount;
            }

            @Override
            public Iterator<Pair<Integer, Label>> iterator() {
                return idLabelNameMap.entrySet().stream()
                    .map(entry -> new Pair<>(entry.getKey(), labels.get(entry.getValue())))
                    .iterator();
            }

            @Override
            public Set<Label> getDomain() {
                return Collections.unmodifiableSet(new HashSet<>(labels.values()));
            }

            @Override
            public int size() {
                return labelNameCounts.size();
            }

            @Override
            public int getUnknownCount() {
                return unknownCount;
            }

            @Override
            public ImmutableOutputInfo<Label> generateImmutableOutputInfo() {
                return this;
            }

            @Override
            public MutableOutputInfo<Label> generateMutableOutputInfo() {
                throw new UnsupportedOperationException();
            }

            @Override
            public OutputInfo<Label> copy() {
                // we're immutable, no need to copy
                return this;
            }

            @Override
            public String toReadableString() {
                StringBuilder builder = new StringBuilder();
                for (Map.Entry<String, MutableLong> e : labelNameCounts.entrySet()) {
                    if (builder.length() > 0) {
                        builder.append(", ");
                    }
                    builder.append('(');
                    builder.append(labelNameIDMap.get(e.getKey()));
                    builder.append(',');
                    builder.append(e.getKey());
                    builder.append(',');
                    builder.append(e.getValue().longValue());
                    builder.append(')');
                }
                return builder.toString();
            }

            @Override
            public Iterable<Pair<String, Long>> outputCountsIterable() {
                return outputCountsIterable;
            }

            // FIXME equals, hashCode, readObject
        };
    }
}

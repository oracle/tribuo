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

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.MutableLabelInfo;
import org.tribuo.classification.evaluation.ClassifierEvaluation;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.classification.evaluation.LabelConfusionMatrix;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.impl.ListExample;
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

        System.out.println("new toString()");
        System.out.println(mcm);

        System.out.println("\npredictions");
        evaluation.getPredictions().forEach(System.out::println);

        final List<Prediction<MultiLabel>> predictions = evaluation.getPredictions();
        System.out.println("\nsingleLabelConfusionMatrix");
        System.out.println(singleLabelConfusionMatrix(predictions));
    }

    public static LabelConfusionMatrix singleLabelConfusionMatrix(final List<Prediction<MultiLabel>> predictions) {
        final List<Prediction<Label>> singleLabelPredictions = mkSingleLabelPredictions(predictions);
        ImmutableOutputInfo<Label> domain = mkDomain(singleLabelPredictions);
        LabelConfusionMatrix cm = new LabelConfusionMatrix(domain, singleLabelPredictions);
        return cm;
    }

    public static List<Prediction<Label>> mkSingleLabelPredictions(List<Prediction<MultiLabel>> predictions) {
        return mkSingleLabelPredictions(predictions, false);
    }

    public static List<Prediction<Label>> mkSingleLabelPredictions(List<Prediction<MultiLabel>> predictions,
        final boolean falseNegativeHeuristic) {
        return predictions.stream()
          .flatMap(p -> {
              final Set<Label> trueLabels = p.getExample().getOutput().getLabelSet();
              final Set<Label> predicted = p.getOutput().getLabelSet();
              // intersection(trueLabels, predicted) = true positives
              // predicted - trueLabels = false positives
              // trueLabels - predicted = false negatives
              return Stream.concat(predicted.stream().map(pred -> {
                  if (trueLabels.contains(pred)) {
                      return mkPrediction(pred.getLabel(), pred.getLabel());
                  } else if (trueLabels.size() == 1) {
                      return mkPrediction(trueLabels.iterator().next().getLabel(), pred.getLabel());
                  } else {
                      // arbitrarily pick first trueLabel
                      return mkPrediction(trueLabels.iterator().next().getLabel(), pred.getLabel());
                  }
              }),
              !falseNegativeHeuristic ? Stream.of() :
              // partially represent false negatives by calling them false positives tied to some predicted label if there is one
                  trueLabels.stream().filter(t -> !predicted.contains(t)).flatMap(fnTrueLabel -> {
                  if (predicted.isEmpty()) {
                      // nothing to pin this on
                      return Stream.of();
                  } else if (predicted.size() == 1) {
                      return Stream.of(mkPrediction(fnTrueLabel.getLabel(), predicted.iterator().next().getLabel()));
                  } else {
                      // arbitrarily pick first predicted label
                      return Stream.of(mkPrediction(fnTrueLabel.getLabel(), predicted.iterator().next().getLabel()));
                  }
              })
              );
          }).collect(Collectors.toList());
    }

    // FIXME HACK copied from Classification/Core/src/test/java/org/tribuo/classification/Utils.java

    public static Prediction<Label> mkPrediction(String trueVal, String predVal) {
        LabelFactory factory = new LabelFactory();
        Example<Label> example = new ListExample<>(factory.generateOutput(trueVal));
        example.add(new Feature("noop", 1d));
        Prediction<Label> prediction = new Prediction<>(factory.generateOutput(predVal), 0, example);
        return prediction;
    }

    public static ImmutableOutputInfo<Label> mkDomain(List<Prediction<Label>> predictions) {
      // MutableLabelInfo info = new MutableLabelInfo();
      // FIXME hack call package private ctor of MutableLabelInfo
      // TODO just make that public
      final MutableLabelInfo info;
      try {
          Constructor<MutableLabelInfo> ctor = MutableLabelInfo.class.getDeclaredConstructor();
          ctor.setAccessible(true);
          info = ctor.newInstance();
      } catch (NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
          throw new RuntimeException(e);
      }
      for (Prediction<Label> p : predictions) {
          info.observe(p.getExample().getOutput());
          info.observe(p.getOutput()); // TODO? LN added
      }
      return info.generateImmutableOutputInfo();
    }
}

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

package org.tribuo.multilabel.evaluation;

import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.classification.evaluation.ClassifierEvaluation;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.multilabel.baseline.IndependentMultiLabelTrainer;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import static org.tribuo.multilabel.Utils.label;
import static org.tribuo.multilabel.Utils.mkPrediction;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MultiLabelEvaluatorTest {

    @Test
    public void test() {
        List<Prediction<MultiLabel>> predictions = Arrays.asList(
                mkPrediction(label("a"), label("a", "b")),
                mkPrediction(label("c", "b"), label("b")),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("b"), label("c"))
        );
        Dataset<MultiLabel> dataset = mkDataset(predictions);
        Model<MultiLabel> model = new IndependentMultiLabelTrainer(DummyClassifierTrainer.createMostFrequentTrainer()).train(dataset); // noop model
        assertEquals(3, model.getOutputIDInfo().size());
        ClassifierEvaluation<MultiLabel> evaluation = new MultiLabelEvaluator().evaluate(model, predictions, dataset.getProvenance());

        MultiLabel a = label("a");
        MultiLabel b = label("b");
        MultiLabel c = label("c");

        assertEquals(1, evaluation.tp(a));
        assertEquals(0, evaluation.fp(a));
        assertEquals(3, evaluation.tn(a));
        assertEquals(0, evaluation.fn(a));

        assertEquals(2, evaluation.tp(b));
        assertEquals(1, evaluation.fp(b));
        assertEquals(0, evaluation.tn(b));
        assertEquals(1, evaluation.fn(b));

        assertEquals(0, evaluation.tp(c));
        assertEquals(1, evaluation.fp(c));
        assertEquals(2, evaluation.tn(c));
        assertEquals(1, evaluation.fn(c));

        assertEquals(1d, evaluation.precision(a));
        assertEquals(1d, evaluation.recall(a));
        assertEquals(1d, evaluation.f1(a));

        assertEquals(0.6666666666666666, evaluation.precision(b));
        assertEquals(0.6666666666666666, evaluation.recall(b));
        assertEquals(0.6666666666666666, evaluation.f1(b));

        assertEquals(0d, evaluation.precision(c));
        assertEquals(0d, evaluation.recall(c));
        assertEquals(0d, evaluation.f1(c));

        assertEquals(0.44444444444444453, evaluation.balancedErrorRate());
        assertEquals(0.5555555555555555, evaluation.macroAveragedPrecision());
        assertEquals(0.5555555555555555, evaluation.macroAveragedRecall());
        assertEquals(0.5555555555555555, evaluation.macroAveragedF1());
        assertEquals(0.6, evaluation.microAveragedPrecision());
        assertEquals(0.6, evaluation.microAveragedRecall());
        assertEquals(0.6, evaluation.microAveragedF1());
    }

    @Test
    public void testSingleLabel() {
        List<Prediction<MultiLabel>> predictions = Arrays.asList(
                mkPrediction(label("a"), label("a")),
                mkPrediction(label("c"), label("b")),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("b"), label("c"))
        );
        Dataset<MultiLabel> dataset = mkDataset(predictions);
        Model<MultiLabel> model = new IndependentMultiLabelTrainer(DummyClassifierTrainer.createMostFrequentTrainer()).train(dataset); // noop model
        assertEquals(3, model.getOutputIDInfo().size());
        ClassifierEvaluation<MultiLabel> evaluation = new MultiLabelEvaluator()
                .evaluate(model, predictions, dataset.getProvenance());

        MultiLabel a = label("a");
        MultiLabel b = label("b");
        MultiLabel c = label("c");

        assertEquals(1d, evaluation.precision(a));
        assertEquals(1d, evaluation.recall(a));
        assertEquals(1d, evaluation.f1(a));

        assertEquals(0.5, evaluation.precision(b));
        assertEquals(0.5, evaluation.recall(b));
        assertEquals(0.5, evaluation.f1(b));

        assertEquals(0d, evaluation.precision(c));
        assertEquals(0d, evaluation.recall(c));
        assertEquals(0d, evaluation.f1(c));

        assertEquals(0.5, evaluation.balancedErrorRate());
        assertEquals(0.5, evaluation.macroAveragedPrecision());
        assertEquals(0.5, evaluation.macroAveragedRecall());
        assertEquals(0.5, evaluation.macroAveragedF1());
        assertEquals(0.5, evaluation.microAveragedPrecision());
        assertEquals(0.5, evaluation.microAveragedRecall());
        assertEquals(0.5, evaluation.microAveragedF1());
    }


    private static Dataset<MultiLabel> mkDataset(List<Prediction<MultiLabel>> predictions) {
        List<Example<MultiLabel>> examples = predictions
                .stream().map(Prediction::getExample)
                .collect(Collectors.toList());
        DataSource<MultiLabel> src = new DataSource<MultiLabel>() {
            @Override public OutputFactory<MultiLabel> getOutputFactory() { return new MultiLabelFactory(); }
            @Override public DataSourceProvenance getProvenance() { return new SimpleDataSourceProvenance("", getOutputFactory()); }
            @Override public Iterator<Example<MultiLabel>> iterator() { return examples.iterator(); }
        };
        return new MutableDataset<>(src);
    }

}
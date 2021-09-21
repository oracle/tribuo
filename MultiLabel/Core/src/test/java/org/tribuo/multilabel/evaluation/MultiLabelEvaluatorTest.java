/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.multilabel.baseline.IndependentMultiLabelTrainer;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.junit.jupiter.api.Test;
import org.tribuo.util.Util;

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
                // mkPrediction(MultiLabel trueVal, MultiLabel predVal)
                mkPrediction(label("a"), label("a", "b")),
                mkPrediction(label("c", "b", "d"), label("b")),
                mkPrediction(label("a","b"),label("a","b")),
                mkPrediction(label("a","b","c"),label("a")),
                mkPrediction(label("a","b","c"),label("d")),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("c"), label("d","b")),
                mkPrediction(label("b"), label("b", "a")),
                mkPrediction(label("b"), label("b", "c")),
                mkPrediction(label("c"), label("c")),
                mkPrediction(label("a"), label())
        );

        Dataset<MultiLabel> dataset = mkDataset(predictions);
        Model<MultiLabel> model = new IndependentMultiLabelTrainer(DummyClassifierTrainer.createMostFrequentTrainer()).train(dataset); // noop model
        assertEquals(4, model.getOutputIDInfo().size());
        MultiLabelEvaluation evaluation = new MultiLabelEvaluator().evaluate(model, predictions, dataset.getProvenance());

        MultiLabel a = label("a");
        MultiLabel b = label("b");
        MultiLabel c = label("c");
        MultiLabel d = label("d");

        assertEquals(3, evaluation.tp(a));
        assertEquals(1, evaluation.fp(a));
        assertEquals(5, evaluation.tn(a));
        assertEquals(2, evaluation.fn(a));

        assertEquals(5, evaluation.tp(b));
        assertEquals(2, evaluation.fp(b));
        assertEquals(2, evaluation.tn(b));
        assertEquals(2, evaluation.fn(b));

        assertEquals(1, evaluation.tp(c));
        assertEquals(1, evaluation.fp(c));
        assertEquals(5, evaluation.tn(c));
        assertEquals(4, evaluation.fn(c));

        assertEquals(0, evaluation.tp(d));
        assertEquals(2, evaluation.fp(d));
        assertEquals(8, evaluation.tn(d));
        assertEquals(1, evaluation.fn(d));

        assertEquals(0.75, evaluation.precision(a),1e-10);
        assertEquals(0.6, evaluation.recall(a),1e-10);
        assertEquals(0.666666666666666, evaluation.f1(a),1e-10);

        assertEquals(0.714285714285714, evaluation.precision(b),1e-10);
        assertEquals(0.714285714285714, evaluation.recall(b),1e-10);
        assertEquals(0.714285714285714, evaluation.f1(b),1e-10);

        assertEquals(0.5, evaluation.precision(c),1e-10);
        assertEquals(0.2, evaluation.recall(c),1e-10);
        assertEquals(0.285714285714285, evaluation.f1(c),1e-10);

        assertEquals(0d, evaluation.precision(d));
        assertEquals(0d, evaluation.recall(d));
        assertEquals(0d, evaluation.f1(d));

        assertEquals(0.6214285714285714, evaluation.balancedErrorRate(),1e-10);
        assertEquals(0.4910714285714286, evaluation.macroAveragedPrecision(),1e-10);
        assertEquals(0.37857142857142856, evaluation.macroAveragedRecall(),1e-10);
        assertEquals(0.41666666666666663, evaluation.macroAveragedF1(),1e-10);
        assertEquals(0.6, evaluation.microAveragedPrecision(),1e-10);
        assertEquals(0.5, evaluation.microAveragedRecall(),1e-10);
        assertEquals(0.5454545454545454, evaluation.microAveragedF1(),1e-10);

        double[] jaccard = new double[]{0.5,0.333333333,1,0.333333333,0,1,0,0.5,0.5,1,0};
        double jaccardScore = Util.mean(jaccard);
        assertEquals(jaccardScore, evaluation.jaccardScore(),1e-10);
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
        MultiLabelEvaluation evaluation = new MultiLabelEvaluator()
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

        assertEquals(0.5, evaluation.jaccardScore());
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
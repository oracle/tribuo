/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.ensemble;

import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.test.Helpers;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class VotingCombinerTest {

    private static final LabelFactory factory = new LabelFactory();

    private static final VotingCombiner votingCombiner = new VotingCombiner();

    private static final Label purple = new Label("PURPLE");
    private static final Label monkey = new Label("MONKEY");
    private static final Label dishwasher = new Label("DISHWASHER");
    private static final Label pidgeon = new Label("PIDGEON");
    private static final Label dryer = new Label("DRYER");

    private static final Trainer<Label> purpleTrainer = DummyClassifierTrainer.createConstantTrainer(purple.getLabel());
    private static final Trainer<Label> monkeyTrainer = DummyClassifierTrainer.createConstantTrainer(monkey.getLabel());
    private static final Trainer<Label> dishwasherTrainer = DummyClassifierTrainer.createConstantTrainer(dishwasher.getLabel());
    private static final Trainer<Label> pidgeonTrainer = DummyClassifierTrainer.createConstantTrainer(pidgeon.getLabel());
    private static final Trainer<Label> dryerTrainer = DummyClassifierTrainer.createConstantTrainer(dryer.getLabel());

    private static Dataset<Label> pmdDataset() {
        String[] featureNames = new String[]{"X_0","X_1"};
        double[] featureValues = new double[]{1.0,2.0};

        List<Example<Label>> list = new ArrayList<>();

        list.add(new ArrayExample<>(purple,featureNames,featureValues));
        list.add(new ArrayExample<>(monkey,featureNames,featureValues));
        list.add(new ArrayExample<>(dishwasher,featureNames,featureValues));

        ListDataSource<Label> source = new ListDataSource<>(list,factory,new SimpleDataSourceProvenance("first-test", OffsetDateTime.now(),factory));
        return new MutableDataset<>(source);
    }

    private static Dataset<Label> ppdDataset() {
        String[] featureNames = new String[]{"X_0","X_1"};
        double[] featureValues = new double[]{1.0,2.0};

        List<Example<Label>> list = new ArrayList<>();

        list.add(new ArrayExample<>(purple,featureNames,featureValues));
        list.add(new ArrayExample<>(pidgeon,featureNames,featureValues));
        list.add(new ArrayExample<>(dryer,featureNames,featureValues));

        ListDataSource<Label> source = new ListDataSource<>(list,factory,new SimpleDataSourceProvenance("first-test", OffsetDateTime.now(),factory));
        return new MutableDataset<>(source);
    }

    @Test
    public void votingCombinerTest() {
        Dataset<Label> pmd = pmdDataset();

        Model<Label> purpleModel = purpleTrainer.train(pmd);
        Model<Label> monkeyModel = monkeyTrainer.train(pmd);
        Model<Label> dishwasherModel = dishwasherTrainer.train(pmd);

        List<Model<Label>> modelList = new ArrayList<>();

        Example<Label> testExample = new ArrayExample<>(purple,new String[]{"X_0","X_1","X_2"}, new double[]{1,2,3});

        // Ties are broken with the first prediction
        modelList.add(purpleModel);
        modelList.add(monkeyModel);
        modelList.add(dishwasherModel);
        WeightedEnsembleModel<Label> ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("monkey",modelList,votingCombiner);
        assertEquals(3,ensemble.getNumModels());
        Prediction<Label> prediction = ensemble.predict(testExample);
        assertEquals(prediction.getOutput(),ensemble.getOutputIDInfo().getOutput(0));
        modelList.clear();

        // Combiner predicts the majority vote
        modelList.add(purpleModel);
        modelList.add(purpleModel);
        modelList.add(dishwasherModel);
        ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("purple",modelList,votingCombiner);
        assertEquals(3,ensemble.getNumModels());
        prediction = ensemble.predict(testExample);
        assertEquals(prediction.getOutput(),purple);
        modelList.clear();

        // Weights affect the vote
        modelList.add(purpleModel);
        modelList.add(monkeyModel);
        modelList.add(dishwasherModel);
        ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("dishwasher",modelList,votingCombiner,new float[]{1,1,3});
        assertEquals(3,ensemble.getNumModels());
        prediction = ensemble.predict(testExample);
        assertEquals(prediction.getOutput(),dishwasher);
        modelList.clear();

        Helpers.testModelSerialization(ensemble,Label.class);
    }

    @Test
    public void existingModelEnsembleTest() {
        Dataset<Label> pmd = pmdDataset();
        Dataset<Label> ppd = ppdDataset();

        Model<Label> purpleModel = purpleTrainer.train(pmd);
        Model<Label> monkeyModel = monkeyTrainer.train(pmd);
        Model<Label> dishwasherModel = dishwasherTrainer.train(pmd);
        Model<Label> pidgeonModel = pidgeonTrainer.train(ppd);

        List<Model<Label>> modelList = new ArrayList<>();

        // Empty list throws
        assertThrows(IllegalArgumentException.class, () -> WeightedEnsembleModel.createEnsembleFromExistingModels("empty",modelList,votingCombiner));

        // Less than 2 members throws
        modelList.add(purpleModel);
        assertThrows(IllegalArgumentException.class, () -> WeightedEnsembleModel.createEnsembleFromExistingModels("one-member",modelList,votingCombiner));
        modelList.clear();

        // Mismatch between models and weights throws
        modelList.add(purpleModel);
        modelList.add(monkeyModel);
        assertThrows(IllegalArgumentException.class, () -> WeightedEnsembleModel.createEnsembleFromExistingModels("model-weight-mismatch",modelList,votingCombiner,new float[5]));
        modelList.clear();

        // Mismatch between model output dimensions throws
        modelList.add(purpleModel);
        modelList.add(pidgeonModel);
        assertThrows(IllegalArgumentException.class, () -> WeightedEnsembleModel.createEnsembleFromExistingModels("output-mismatch",modelList,votingCombiner));
        modelList.clear();

        // Created an ensemble
        modelList.add(purpleModel);
        modelList.add(monkeyModel);
        modelList.add(dishwasherModel);
        WeightedEnsembleModel<Label> ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("valid",modelList,votingCombiner);
        assertEquals(3,ensemble.getNumModels());
    }

}

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

package org.tribuo.regression.ensemble;

import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.baseline.DummyRegressionTrainer;
import org.tribuo.test.Helpers;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class AveragingCombinerTest {

    private static final RegressionFactory factory = new RegressionFactory();

    private static final AveragingCombiner averagingCombiner = new AveragingCombiner();

    private static final String[] abc = new String[]{"a","b","c"};
    private static final String[] xyz = new String[]{"x","y","z"};

    private static final Trainer<Regressor> fiveTrainer = DummyRegressionTrainer.createConstantTrainer(5);
    private static final Trainer<Regressor> threeTrainer = DummyRegressionTrainer.createConstantTrainer(3);
    private static final Trainer<Regressor> oneTrainer = DummyRegressionTrainer.createConstantTrainer(1);

    private static Dataset<Regressor> abcDataset() {
        String[] featureNames = new String[]{"X_0","X_1"};
        double[] featureValues = new double[]{1.0,2.0};

        List<Example<Regressor>> list = new ArrayList<>();

        list.add(new ArrayExample<>(new Regressor(abc,new double[]{1,2,3}),featureNames,featureValues));
        list.add(new ArrayExample<>(new Regressor(abc,new double[]{4,5,6}),featureNames,featureValues));
        list.add(new ArrayExample<>(new Regressor(abc,new double[]{-4,-5,-6}),featureNames,featureValues));

        ListDataSource<Regressor> source = new ListDataSource<>(list,factory,new SimpleDataSourceProvenance("first-test", OffsetDateTime.now(),factory));
        return new MutableDataset<>(source);
    }

    private static Dataset<Regressor> xyzDataset() {
        String[] featureNames = new String[]{"X_0","X_1"};
        double[] featureValues = new double[]{1.0,2.0};

        List<Example<Regressor>> list = new ArrayList<>();

        list.add(new ArrayExample<>(new Regressor(xyz,new double[]{3,2,1}),featureNames,featureValues));
        list.add(new ArrayExample<>(new Regressor(xyz,new double[]{6,5,-4}),featureNames,featureValues));
        list.add(new ArrayExample<>(new Regressor(xyz,new double[]{-6,-5,-4}),featureNames,featureValues));

        ListDataSource<Regressor> source = new ListDataSource<>(list,factory,new SimpleDataSourceProvenance("first-test", OffsetDateTime.now(),factory));
        return new MutableDataset<>(source);
    }

    @Test
    public void averagingCombinerTest() {
        Dataset<Regressor> abcDataset = abcDataset();

        Model<Regressor> fiveModel = fiveTrainer.train(abcDataset);
        Model<Regressor> threeModel = threeTrainer.train(abcDataset);
        Model<Regressor> oneModel = oneTrainer.train(abcDataset);

        List<Model<Regressor>> modelList = new ArrayList<>();

        Example<Regressor> testExample = new ArrayExample<>(new Regressor(abc,new double[]{-1,-1,-1}),new String[]{"X_0","X_1","X_2"}, new double[]{1,2,3});

        // Combiner predicts the average
        modelList.add(fiveModel);
        modelList.add(threeModel);
        modelList.add(oneModel);
        WeightedEnsembleModel<Regressor> ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("average",modelList, averagingCombiner);
        assertEquals(3,ensemble.getNumModels());
        Prediction<Regressor> prediction = ensemble.predict(testExample);
        Regressor target = new Regressor(abc,new double[]{3,3,3});
        assertArrayEquals(prediction.getOutput().getValues(),target.getValues());
        modelList.clear();

        // Weights affect the averaging
        modelList.add(fiveModel);
        modelList.add(threeModel);
        modelList.add(oneModel);
        ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("weighted",modelList, averagingCombiner,new float[]{3,1,1});
        assertEquals(3,ensemble.getNumModels());
        prediction = ensemble.predict(testExample);
        target = new Regressor(abc,new double[]{3.8,3.8,3.8});
        assertArrayEquals(prediction.getOutput().getValues(),target.getValues());
        modelList.clear();

        Helpers.testModelSerialization(ensemble,Regressor.class);
    }

    @Test
    public void existingModelEnsembleTest() {
        Dataset<Regressor> abcDataset = abcDataset();
        Dataset<Regressor> xyzDataset = xyzDataset();

        Model<Regressor> fiveModel = fiveTrainer.train(abcDataset);
        Model<Regressor> oneModel = oneTrainer.train(abcDataset);
        Model<Regressor> fiveXYZModel = fiveTrainer.train(xyzDataset);

        List<Model<Regressor>> modelList = new ArrayList<>();

        // Mismatch between model output dimensions throws
        modelList.add(fiveModel);
        modelList.add(fiveXYZModel);
        assertThrows(IllegalArgumentException.class, () -> WeightedEnsembleModel.createEnsembleFromExistingModels("invalid-dimensions",modelList, averagingCombiner));
        modelList.clear();

        // Created an ensemble
        modelList.add(fiveModel);
        modelList.add(oneModel);
        WeightedEnsembleModel<Regressor> ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("valid",modelList, averagingCombiner);
        assertEquals(2,ensemble.getNumModels());
    }

}

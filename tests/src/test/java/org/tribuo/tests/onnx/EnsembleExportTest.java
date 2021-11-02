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

package org.tribuo.tests.onnx;

import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.ensemble.FullyWeightedVotingCombiner;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.example.NoisyInterlockingCrescentsDataSource;
import org.tribuo.classification.libsvm.LibSVMClassificationModel;
import org.tribuo.classification.libsvm.LibSVMClassificationTrainer;
import org.tribuo.classification.libsvm.SVMClassificationType;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.common.sgd.AbstractFMModel;
import org.tribuo.common.sgd.AbstractFMTrainer;
import org.tribuo.common.sgd.AbstractSGDTrainer;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.ensemble.EnsembleModel;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.ensemble.MultiLabelVotingCombiner;
import org.tribuo.multilabel.example.MultiLabelGaussianDataSource;
import org.tribuo.multilabel.sgd.fm.FMMultiLabelTrainer;
import org.tribuo.multilabel.sgd.objectives.BinaryCrossEntropy;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.example.NonlinearGaussianDataSource;
import org.tribuo.regression.sgd.fm.FMRegressionTrainer;
import org.tribuo.regression.sgd.objectives.SquaredLoss;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

public class EnsembleExportTest {

    // Classification combiners
    private static final VotingCombiner VOTING = new VotingCombiner();
    private static final FullyWeightedVotingCombiner FULL_VOTING = new FullyWeightedVotingCombiner();

    // Regression combiners
    private static final AveragingCombiner AVERAGING = new AveragingCombiner();

    // Multi-label combiners
    private static final MultiLabelVotingCombiner ML_VOTING = new MultiLabelVotingCombiner();

    @BeforeAll
    public static void setup() {
        Class<?>[] classes = new Class<?>[]{
                BaggingTrainer.class,
                AbstractSGDTrainer.class,
                org.tribuo.classification.sgd.linear.LinearSGDTrainer.class,
                AbstractFMTrainer.class,
                FMClassificationTrainer.class
        };
        for (Class<?> c : classes) {
            Logger logger = Logger.getLogger(c.getName());
            logger.setLevel(Level.WARNING);
        }
    }

    @Test
    public void testHomogenousClassificationExport() throws IOException, OrtException {
        // Prep data
        DataSource<Label> trainSource = new NoisyInterlockingCrescentsDataSource(100,1,0.1);
        MutableDataset<Label> train = new MutableDataset<>(trainSource);
        DataSource<Label> testSource = new NoisyInterlockingCrescentsDataSource(100,2,0.1);
        MutableDataset<Label> test = new MutableDataset<>(testSource);

        // Train model
        LogisticRegressionTrainer lr = new LogisticRegressionTrainer();
        BaggingTrainer<Label> t = new BaggingTrainer<>(lr, VOTING,5);
        WeightedEnsembleModel<Label> ensemble = (WeightedEnsembleModel<Label>) t.train(train);

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-bagging-test",".onnx");
        ensemble.saveONNXModel("org.tribuo.ensemble.test",1,onnxFile);

        OnnxTestUtils.onnxLabelComparison(ensemble,onnxFile,test,1e-6);

        onnxFile.toFile().delete();
    }

    @Test
    public void testHeterogeneousClassificationExport() throws IOException, OrtException {
        // Prep data
        DataSource<Label> trainSource = new NoisyInterlockingCrescentsDataSource(100,1,0.1);
        MutableDataset<Label> train = new MutableDataset<>(trainSource);
        DataSource<Label> testSource = new NoisyInterlockingCrescentsDataSource(100,2,0.1);
        MutableDataset<Label> test = new MutableDataset<>(testSource);

        // Train model
        AdaGrad adagrad = new AdaGrad(0.1,0.1);
        LogisticRegressionTrainer lr = new LogisticRegressionTrainer();
        BaggingTrainer<Label> t = new BaggingTrainer<>(lr, VOTING,5);
        EnsembleModel<Label> bagModel = t.train(train);
        FMClassificationTrainer fmT = new FMClassificationTrainer(new LogMulticlass(),adagrad,2,100,1,1L,5,0.1);
        AbstractFMModel<Label> fmModel = fmT.train(train);
        LibSVMClassificationTrainer svmT = new LibSVMClassificationTrainer(new SVMParameters<>(new SVMClassificationType(SVMClassificationType.SVMMode.NU_SVC), KernelType.RBF));
        LibSVMClassificationModel svmModel = (LibSVMClassificationModel) svmT.train(train);
        WeightedEnsembleModel<Label> ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("Bag+FM", Arrays.asList(bagModel,fmModel,svmModel), FULL_VOTING, new float[]{0.3f,0.5f,0.2f});

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-bagging-test",".onnx");
        ensemble.saveONNXModel("org.tribuo.ensemble.test",1,onnxFile);

        OnnxTestUtils.onnxLabelComparison(ensemble,onnxFile,test,1e-6);

        onnxFile.toFile().delete();
    }

    @Test
    public void testHomogenousRegressionExport() throws IOException, OrtException {
        // Prep data
        DataSource<Regressor> trainSource = new NonlinearGaussianDataSource(100,1L);
        MutableDataset<Regressor> train = new MutableDataset<>(trainSource);
        DataSource<Regressor> testSource = new NonlinearGaussianDataSource(100,2L);
        MutableDataset<Regressor> test = new MutableDataset<>(testSource);

        // Train model
        org.tribuo.regression.sgd.linear.LinearSGDTrainer lr = new org.tribuo.regression.sgd.linear.LinearSGDTrainer(new SquaredLoss(), new AdaGrad(0.1,0.1), 2, 1L);
        BaggingTrainer<Regressor> t = new BaggingTrainer<>(lr,AVERAGING,5);
        WeightedEnsembleModel<Regressor> ensemble = (WeightedEnsembleModel<Regressor>) t.train(train);

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-bagging-test",".onnx");
        ensemble.saveONNXModel("org.tribuo.ensemble.test",1,onnxFile);

        OnnxTestUtils.onnxRegressorComparison(ensemble,onnxFile,test,1e-5);

        onnxFile.toFile().delete();
    }

    @Test
    public void testHeterogeneousRegressionExport() throws IOException, OrtException {
        // Prep data
        DataSource<Regressor> trainSource = new NonlinearGaussianDataSource(100,1L);
        MutableDataset<Regressor> train = new MutableDataset<>(trainSource);
        DataSource<Regressor> testSource = new NonlinearGaussianDataSource(100,2L);
        MutableDataset<Regressor> test = new MutableDataset<>(testSource);

        // Train model
        SquaredLoss loss = new SquaredLoss();
        AdaGrad adagrad = new AdaGrad(0.1,0.1);
        org.tribuo.regression.sgd.linear.LinearSGDTrainer lr = new org.tribuo.regression.sgd.linear.LinearSGDTrainer(loss, adagrad, 2, 1L);
        BaggingTrainer<Regressor> t = new BaggingTrainer<>(lr,AVERAGING,5);
        WeightedEnsembleModel<Regressor> bagModel = (WeightedEnsembleModel<Regressor>) t.train(train);
        FMRegressionTrainer fmT = new FMRegressionTrainer(loss,adagrad,2,100,1,1L,5,0.1,true);
        AbstractFMModel<Regressor> fmModel = fmT.train(train);
        WeightedEnsembleModel<Regressor> ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("Bag+FM", Arrays.asList(bagModel,fmModel), AVERAGING, new float[]{0.3f,0.7f});

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-bagging-test",".onnx");
        ensemble.saveONNXModel("org.tribuo.ensemble.test",1,onnxFile);

        OnnxTestUtils.onnxRegressorComparison(ensemble,onnxFile,test,1e-5);

        onnxFile.toFile().delete();
    }

    @Test
    public void testHomogenousMultiLabelExport() throws IOException, OrtException {
        // Prep data
        DataSource<MultiLabel> trainSource = MultiLabelGaussianDataSource.makeDefaultSource(100,1L);
        MutableDataset<MultiLabel> train = new MutableDataset<>(trainSource);
        DataSource<MultiLabel> testSource = MultiLabelGaussianDataSource.makeDefaultSource(100,2L);
        MutableDataset<MultiLabel> test = new MutableDataset<>(testSource);

        // Train model
        BinaryCrossEntropy loss = new BinaryCrossEntropy();
        AdaGrad adagrad = new AdaGrad(0.1,0.1);
        org.tribuo.multilabel.sgd.linear.LinearSGDTrainer lr = new org.tribuo.multilabel.sgd.linear.LinearSGDTrainer(loss,adagrad,3,1000,1,1L);
        BaggingTrainer<MultiLabel> t = new BaggingTrainer<>(lr, ML_VOTING,5);
        WeightedEnsembleModel<MultiLabel> ensemble = (WeightedEnsembleModel<MultiLabel>) t.train(train);

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-bagging-test",".onnx");
        ensemble.saveONNXModel("org.tribuo.ensemble.test",1,onnxFile);

        OnnxTestUtils.onnxMultiLabelComparison(ensemble,onnxFile,test,1e-6);

        onnxFile.toFile().delete();

    }

    @Test
    public void testHeterogeneousMultiLabelExport() throws IOException, OrtException {
        // Prep data
        DataSource<MultiLabel> trainSource = MultiLabelGaussianDataSource.makeDefaultSource(100,1L);
        MutableDataset<MultiLabel> train = new MutableDataset<>(trainSource);
        DataSource<MultiLabel> testSource = MultiLabelGaussianDataSource.makeDefaultSource(100,2L);
        MutableDataset<MultiLabel> test = new MutableDataset<>(testSource);

        // Train model
        BinaryCrossEntropy loss = new BinaryCrossEntropy();
        AdaGrad adagrad = new AdaGrad(0.1,0.1);
        org.tribuo.multilabel.sgd.linear.LinearSGDTrainer lr = new org.tribuo.multilabel.sgd.linear.LinearSGDTrainer(loss,adagrad,3,1000,1,1L);
        BaggingTrainer<MultiLabel> t = new BaggingTrainer<>(lr, ML_VOTING,5);
        EnsembleModel<MultiLabel> bagModel = t.train(train);
        FMMultiLabelTrainer fmT = new FMMultiLabelTrainer(loss,adagrad,2,100,1,1L,5,0.1);
        AbstractFMModel<MultiLabel> fmModel = fmT.train(train);
        WeightedEnsembleModel<MultiLabel> ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("Bag+FM", Arrays.asList(bagModel,fmModel), ML_VOTING, new float[]{0.3f,0.7f});

        // Write out model
        Path onnxFile = Files.createTempFile("tribuo-bagging-test",".onnx");
        ensemble.saveONNXModel("org.tribuo.ensemble.test",1,onnxFile);

        OnnxTestUtils.onnxMultiLabelComparison(ensemble,onnxFile,test,1e-6);

        onnxFile.toFile().delete();
    }

}

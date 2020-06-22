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

package org.tribuo.classification.explanations.lime;

import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import org.tribuo.ImmutableDataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.SparseTrainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.datasource.LibSVMDataSource;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.rtree.CARTJointRegressionTrainer;

import java.io.File;
import java.io.IOException;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * An example which runs MNIST through LIME. Expects the path to libsvm format MNIST training data as the first argument
 * and the path to libsvm format MNIST test data as the second argument.
 */
public class MNISTDemo {
    private static final Logger logger = Logger.getLogger(MNISTDemo.class.getName());

    private static final int numFeatures = 10;

    public static void main(String[] args) throws IOException {
        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        LabelFactory labelFactory = new LabelFactory();
        logger.info("Loading training data.");
        //
        // Load the libsvm text-based data format.
        LibSVMDataSource<Label> trainSource = new LibSVMDataSource<>(new File(args[0]).toPath(),labelFactory);
        MutableDataset<Label> train = new MutableDataset<>(trainSource);
        boolean zeroIndexed = trainSource.isZeroIndexed();
        int maxFeatureID = trainSource.getMaxFeatureID();
        logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
        logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
        LibSVMDataSource<Label> testSource = new LibSVMDataSource<>(new File(args[1]).toPath(),labelFactory,zeroIndexed,maxFeatureID);
        ImmutableDataset<Label> test = new ImmutableDataset<>(testSource,train.getFeatureIDMap(),train.getOutputIDInfo(),false);
        logger.info(String.format("Loaded %d testing examples", test.size()));

        //public XGBoostClassificationTrainer(int numTrees, double eta, double gamma, int maxDepth, double minChildWeight, double subsample, double featureSubsample, double lambda, double alpha, long seed) {
        XGBoostClassificationTrainer trainer = new XGBoostClassificationTrainer(50);
        Model<Label> model = trainer.train(train);
        logger.info("Finished training model");

        //SparseTrainer<Regressor> limeTrainer = new LARSLassoTrainer(numFeatures);
        SparseTrainer<Regressor> limeTrainer = new CARTJointRegressionTrainer((int)(Math.log(numFeatures)/Math.log(2)));

        LIMEBase lime = new LIMEBase(new SplittableRandom(1),model,limeTrainer,200);

        LIMEExplanation e = lime.explain(test.getData().get(0));

        logger.info("Finished lime");
        logger.info("Explanation = " + e.toString());

        LabelEvaluator labelEvaluator = new LabelEvaluator();
        LabelEvaluation evaluation = labelEvaluator.evaluate(model,test);
        logger.info("Finished evaluating model");
        System.out.println(labelEvaluator.toString());
        System.out.println();
        ConfusionMatrix<Label> matrix = evaluation.getConfusionMatrix();
        System.out.println(matrix.toString());
    }

}

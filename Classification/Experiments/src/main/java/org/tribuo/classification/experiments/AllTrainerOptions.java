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

package org.tribuo.classification.experiments;

import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.Trainer;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.classification.Label;
import org.tribuo.classification.dtree.CARTClassificationOptions;
import org.tribuo.classification.ensemble.ClassificationEnsembleOptions;
import org.tribuo.classification.liblinear.LibLinearOptions;
import org.tribuo.classification.libsvm.LibSVMOptions;
import org.tribuo.classification.mnb.MultinomialNaiveBayesOptions;
import org.tribuo.classification.sgd.kernel.KernelSVMOptions;
import org.tribuo.classification.sgd.linear.LinearSGDOptions;
import org.tribuo.classification.xgboost.XGBoostOptions;
import org.tribuo.common.nearest.KNNClassifierOptions;
import org.tribuo.hash.HashingOptions;
import org.tribuo.hash.HashingOptions.ModelHashingType;

import java.util.logging.Logger;

/**
 * Aggregates all the classification algorithms.
 */
public class AllTrainerOptions implements ClassificationOptions<Trainer<Label>> {
    private static final Logger logger = Logger.getLogger(AllTrainerOptions.class.getName());

    public enum AlgorithmType {
        CART, KNN, LIBLINEAR, LIBSVM, MNB, SGD_KERNEL, SGD_LINEAR, XGBOOST //RANDOM_FOREST,
    }

    @Option(longName = "algorithm", usage = "Type of learner (or base learner). Defaults to SGD_LINEAR.")
    public AlgorithmType algorithm = AlgorithmType.SGD_LINEAR;

    public CARTClassificationOptions cartOptions;
    public KNNClassifierOptions knnOptions;
    public LibLinearOptions liblinearOptions;
    public LibSVMOptions libsvmOptions;
    public MultinomialNaiveBayesOptions mnbOptions;
    public KernelSVMOptions kernelSVMOptions;
    public LinearSGDOptions linearSGDOptions;
    public XGBoostOptions xgBoostOptions;

    public ClassificationEnsembleOptions ensemble;
    public HashingOptions hashingOptions;

    @Override
    public Trainer<Label> getTrainer() {
        Trainer<Label> trainer;
        logger.info("Using " + algorithm);
        switch (algorithm) {
            case CART:
                trainer = cartOptions.getTrainer();
                break;
            case KNN:
                trainer = knnOptions.getTrainer();
                break;
            case LIBLINEAR:
                trainer = liblinearOptions.getTrainer();
                break;
            case LIBSVM:
                trainer = libsvmOptions.getTrainer();
                break;
            case MNB:
                trainer = mnbOptions.getTrainer();
                break;
            case SGD_KERNEL:
                trainer = kernelSVMOptions.getTrainer();
                break;
            case SGD_LINEAR:
                trainer = linearSGDOptions.getTrainer();
                break;
            case XGBOOST:
                trainer = xgBoostOptions.getTrainer();
                break;
            default:
                throw new IllegalArgumentException("Unknown classifier " + algorithm);
        }

        if ((ensemble.ensembleSize > 0) && (ensemble.type != null)) {
            switch (algorithm) {
                case XGBOOST:
                    throw new IllegalArgumentException(
                            "Not allowed to ensemble XGBoost models. Why ensemble an ensemble?");
                default:
                    trainer = ensemble.wrapTrainer(trainer);
                    break;
            }
        }

        if (hashingOptions.modelHashingAlgorithm != ModelHashingType.NONE) {
            trainer = hashingOptions.getHashedTrainer(trainer);
        }
        logger.info("Trainer description " + trainer.toString());
        return trainer;
    }

}

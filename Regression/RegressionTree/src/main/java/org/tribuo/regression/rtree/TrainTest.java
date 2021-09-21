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

package org.tribuo.regression.rtree;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.data.DataOptions;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.rtree.impurity.MeanAbsoluteError;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.regression.rtree.impurity.RegressorImpurity;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run a regression tree for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Impurity function.
     */
    public enum ImpurityType {
        /**
         * Use {@link MeanSquaredError}.
         */
        MSE,
        /**
         * Use {@link MeanAbsoluteError}.
         */
        MAE
    }

    /**
     * Type of tree trainer.
     */
    public enum TreeType {
        /**
         * Creates a {@link CARTRegressionTrainer} which treats
         * each regression dimension independently.
         */
        CART_INDEPENDENT,
        /**
         * Creates a {@link CARTJointRegressionTrainer} which
         * jointly minimises the impurity across all output dimensions.
         */
        CART_JOINT
    }

    /**
     * Command line options.
     */
    public static class RegressionTreeOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests a CART regression model on the specified datasets.";
        }

        /**
         * The data loading options.
         */
        public DataOptions general;
        /**
         * Character to split the CSV response on to generate multiple regression dimensions. Defaults to ':'.
         */
        @Option(longName = "csv-response-split-char", usage = "Character to split the CSV response on to generate multiple regression dimensions. Defaults to ':'.")
        public char splitChar = ':';
        /**
         * Maximum depth in the decision tree.
         */
        @Option(charName = 'd', longName = "max-depth", usage = "Maximum depth in the decision tree.")
        public int depth = 6;
        /**
         * Fraction of features in split.
         */
        @Option(charName = 'e', longName = "split-fraction", usage = "Fraction of features in split.")
        public float fraction = 1.0f;
        /**
         * Minimum child weight.
         */
        @Option(charName = 'm', longName = "min-child-weight", usage = "Minimum child weight.")
        public float minChildWeight = 5.0f;
        /**
         * Minimumum decrease in impurity required in order for the node to be split.
         */
        @Option(charName = 'p', longName = "min-impurity-decrease", usage = "Minimumum decrease in impurity required in order for the node to be split.")
        public float minImpurityDecrease = 0.0f;
        /**
         * Whether to choose split points for features at random.
         */
        @Option(charName = 'r', longName = "use-random-split-points", usage = "Whether to choose split points for features at random.")
        public boolean useRandomSplitPoints = false;
        /**
         * Normalize the leaf outputs so each leaf sums to 1.0.
         */
        @Option(charName = 'n', longName = "normalize", usage = "Normalize the leaf outputs so each leaf sums to 1.0.")
        public boolean normalize = false;
        /**
         * Impurity measure to use. Defaults to MSE.
         */
        @Option(charName = 'i', longName = "impurity", usage = "Impurity measure to use. Defaults to MSE.")
        public ImpurityType impurityType = ImpurityType.MSE;
        /**
         * Tree type.
         */
        @Option(charName = 't', longName = "tree-type", usage = "Tree type.")
        public TreeType treeType = TreeType.CART_INDEPENDENT;
        /**
         * Prints the decision tree.
         */
        @Option(longName = "print-tree", usage = "Prints the decision tree.")
        public boolean printTree;
    }

    /**
     * Runs a TrainTest CLI.
     * @param args the command line arguments
     * @throws IOException if there is any error reading the examples.
     */
    public static void main(String[] args) throws IOException {

        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        RegressionTreeOptions o = new RegressionTreeOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        RegressionFactory factory = new RegressionFactory(o.splitChar);

        Pair<Dataset<Regressor>,Dataset<Regressor>> data = o.general.load(factory);
        Dataset<Regressor> train = data.getA();
        Dataset<Regressor> test = data.getB();

        RegressorImpurity impurity;
        switch (o.impurityType) {
            case MAE:
                impurity = new MeanAbsoluteError();
                break;
            case MSE:
                impurity = new MeanSquaredError();
                break;
            default:
                logger.severe("unknown impurity type " + o.impurityType);
                return;
        }

        if (o.general.trainingPath == null || o.general.testingPath == null) {
            logger.info(cm.usage());
            return;
        }

        SparseTrainer<Regressor> trainer;
        switch (o.treeType) {
            case CART_INDEPENDENT:
                    trainer = new CARTRegressionTrainer(o.depth, o.minChildWeight,o.minImpurityDecrease,o.fraction, o.useRandomSplitPoints, impurity,
                            o.general.seed);
                break;
            case CART_JOINT:
                    trainer = new CARTJointRegressionTrainer(o.depth, o.minChildWeight, o.minImpurityDecrease, o.fraction, o.useRandomSplitPoints,
                            impurity, o.normalize, o.general.seed);
                break;
            default:
                logger.severe("unknown tree type " + o.treeType);
                return;
        }

        logger.info("Training using " + trainer.toString());

        final long trainStart = System.currentTimeMillis();
        SparseModel<Regressor> model = trainer.train(train);
        final long trainStop = System.currentTimeMillis();

        logger.info("Finished training regressor " + Util.formatDuration(trainStart,trainStop));

        if (o.printTree) {
            logger.info(model.toString());
        }

        logger.info("Selected features: " + model.getActiveFeatures());
        final long testStart = System.currentTimeMillis();
        RegressionEvaluation evaluation = factory.getEvaluator().evaluate(model,test);
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart,testStop));
        System.out.println(evaluation.toString());

        if (o.general.outputPath != null) {
            o.general.saveModel(model);
        }
    }
}

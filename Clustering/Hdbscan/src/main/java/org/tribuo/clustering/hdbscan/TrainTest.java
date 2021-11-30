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

package org.tribuo.clustering.hdbscan;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.clustering.evaluation.ClusteringEvaluation;
import org.tribuo.data.DataOptions;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run a HDBSCAN* clustering model for a standard dataset.
 */
final public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Options for the HDBSCAN* CLI.
     */
    public static class HdbscanOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and evaluates a HDBSCAN* model on the specified dataset.";
        }

        /**
         * The data loading options.
         */
        public DataOptions general;

        /**
         * The minimum number of points required to form a cluster.
         */
        @Option(longName = "minimum-cluster-size", usage = "The minimum number of points required to form a cluster. Defaults to 5.")
        public int minClusterSize = 5;

        /**
         * Distance function in HDBSCAN*. Defaults to EUCLIDEAN.
         */
        @Option(longName = "distance-function", usage = "Distance function to use for various distance calculations. Defaults to EUCLIDEAN.")
        public HdbscanTrainer.Distance distanceType = HdbscanTrainer.Distance.EUCLIDEAN;

        /**
         * The number of nearest-neighbors to use in the initial density approximation.
         */
        @Option(longName = "k-nearest-neighbors", usage = "The number of nearest-neighbors to use in the initial density approximation. " +
            "The value includes the point itself. Defaults to 5.")
        public int k = 5;

        /**
         * Number of threads to use for training the hdbscan model. Defaults to 2.
         */
        @Option(longName = "hdbscan-num-threads", usage = "Number of threads to use for training the hdbscan model. Defaults to 2.")
        public int numThreads = 2;
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

        HdbscanOptions o = new HdbscanOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.general.trainingPath == null) {
            logger.info(cm.usage());
            return;
        }

        ClusteringFactory factory = new ClusteringFactory();

        Pair<Dataset<ClusterID>,Dataset<ClusterID>> data = o.general.load(factory);
        Dataset<ClusterID> train = data.getA();

        HdbscanTrainer trainer = new HdbscanTrainer(o.minClusterSize, o.distanceType, o.k, o.numThreads);
        Model<ClusterID> model = trainer.train(train);
        logger.info("Finished training model");
        ClusteringEvaluation evaluation = factory.getEvaluator().evaluate(model,train);
        logger.info("Finished evaluating model");
        System.out.println("Normalized MI = " + evaluation.normalizedMI());
        System.out.println("Adjusted MI = " + evaluation.adjustedMI());

        if (o.general.outputPath != null) {
            o.general.saveModel(model);
        }
    }
}

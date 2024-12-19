/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.clustering.gmm;

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
import org.tribuo.clustering.gmm.GMMTrainer.Initialisation;
import org.tribuo.data.DataOptions;
import org.tribuo.math.distributions.MultivariateNormalDistribution;

import java.io.IOException;
import java.util.logging.Logger;


/**
 * Build and run a Gaussian Mixture model for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Options for the GMM CLI.
     */
    public static class GMMOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and evaluates a Gaussian Mixture model on the specified dataset.";
        }

        /**
         * The data loading options.
         */
        public DataOptions general;

        /**
         * Number of clusters to infer.
         */
        @Option(charName = 'n', longName = "num-clusters", usage = "Number of clusters to infer.")
        public int centroids = 5;
        /**
         * Maximum number of iterations.
         */
        @Option(charName = 'i', longName = "iterations", usage = "Maximum number of iterations.")
        public int iterations = 10;
        /**
         * The covariance type of the gaussians.
         */
        @Option(charName = 'v', longName = "covariance-type", usage = "Set the covariance type.")
        public MultivariateNormalDistribution.CovarianceType covarianceType = MultivariateNormalDistribution.CovarianceType.DIAGONAL;
        /**
         * Type of initialisation to use for centroids.
         */
        @Option(charName = 's', longName = "initialisation", usage = "Type of initialisation to use for centroids.")
        public Initialisation initialisation = Initialisation.RANDOM;
        /**
         * Convergence tolerance to terminate EM early.
         */
        @Option(longName = "tolerance", usage = "The convergence threshold.")
        public double tolerance = 1e-3;
        /**
         * Number of threads to use (range (1, num hw threads)).
         */
        @Option(charName = 't', longName = "num-threads", usage = "Number of threads to use (range (1, num hw threads)).")
        public int numThreads = 4;
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

        GMMOptions o = new GMMOptions();
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

        //public GMMTrainer(int centroids, int iterations, DistanceType distType, int numThreads, int seed)
        GMMTrainer trainer = new GMMTrainer(o.centroids,o.iterations,
                o.covarianceType,o.initialisation,o.tolerance,o.numThreads,o.general.seed);
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

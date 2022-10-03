/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sgd.fm;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.TrainTestHelper;
import org.tribuo.classification.ensemble.ClassificationEnsembleOptions;
import org.tribuo.data.DataOptions;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run a classifier for a standard dataset using FMClassificationTrainer.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Command line options.
     */
    public static class TrainTestOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests a linear SGD model on the specified datasets.";
        }

        /**
         * The data loading options.
         */
        public DataOptions general;
        /**
         * The factorization machine trainer options.
         */
        public FMClassificationOptions trainerOptions;
        /**
         * The ensemble options.
         */
        public ClassificationEnsembleOptions ensembleOptions;
    }

    /**
     * Runs a TrainTest CLI.
     * @param args the command line arguments
     * @throws IOException if there is any error reading the examples.
     */
    public static void main(String[] args) throws IOException {
        TrainTestOptions o = new TrainTestOptions();
        try (ConfigurationManager cm = new ConfigurationManager(args,o)) {
            Trainer<Label> trainer = o.trainerOptions.getTrainer();
            trainer = o.ensembleOptions.wrapTrainer(trainer);
            TrainTestHelper.run(cm, o.general, trainer);
        } catch (UsageException e) {
            logger.info(e.getMessage());
        }
    }
}

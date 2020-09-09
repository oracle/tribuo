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

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.TrainTestHelper;
import org.tribuo.data.DataOptions;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run a classifier for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    public static class AllClassificationOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests the specified classifier on the supplied datasets.";
        }

        public AllTrainerOptions trainerOptions;
        public DataOptions general;
    }

    public static void main(String[] args) throws IOException {
        AllClassificationOptions o = new AllClassificationOptions();
        try (ConfigurationManager cm = new ConfigurationManager(args,o)){
            Trainer<Label> trainer = o.trainerOptions.getTrainer();
            TrainTestHelper.run(cm, o.general, trainer);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }
    }
}

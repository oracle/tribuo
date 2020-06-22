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

package org.tribuo.classification.xgboost;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.classification.TrainTestHelper;
import org.tribuo.data.DataOptions;

import java.io.IOException;

/**
 * Build and run an XGBoost classifier for a standard dataset.
 */
public class TrainTest {

    public static class TrainTestOptions implements Options {

        @Override
        public String getOptionsDescription() {
            return "Trains and tests an XGBoost classification model on the specified datasets.";
        }

        public DataOptions general;
        public XGBoostOptions xgboostOptions;
    }

    /**
     * @param args the command line arguments
     * @throws java.io.IOException if there is any error reading the examples.
     */
    public static void main(String[] args) throws IOException {
        TrainTestOptions o = new TrainTestOptions();
        ConfigurationManager cm = new ConfigurationManager(args, o);
        XGBoostClassificationTrainer trainer = o.xgboostOptions.getTrainer();
        TrainTestHelper.run(cm, o.general, trainer);
        cm.close();
    }
}

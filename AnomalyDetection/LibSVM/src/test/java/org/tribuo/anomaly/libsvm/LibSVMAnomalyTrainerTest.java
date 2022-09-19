/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.anomaly.libsvm;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.evaluation.AnomalyEvaluation;
import org.tribuo.anomaly.evaluation.AnomalyEvaluator;
import org.tribuo.anomaly.example.AnomalyDataGenerator;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.SVMParameters;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LibSVMAnomalyTrainerTest {
    private static final Logger logger = Logger.getLogger(LibSVMAnomalyTrainerTest.class.getName());

    @Test
    public void gaussianDataTest() {
        Pair<Dataset<Event>,Dataset<Event>> pair = AnomalyDataGenerator.gaussianAnomaly(1000,0.2);

        SVMParameters<Event> params = new SVMParameters<>(new SVMAnomalyType(SVMAnomalyType.SVMMode.ONE_CLASS), KernelType.RBF);
        params.setGamma(1.0);
        params.setNu(0.1);

        LibSVMAnomalyTrainer trainer = new LibSVMAnomalyTrainer(params);

        LibSVMModel<Event> model = trainer.train(pair.getA());

        AnomalyEvaluator evaluator = new AnomalyEvaluator();

        AnomalyEvaluation evaluation = evaluator.evaluate(model,pair.getB());

        assertEquals(200,evaluation.getTruePositives());
        assertTrue(650 < evaluation.getTrueNegatives());
        assertEquals(0,evaluation.getFalseNegatives());

        String confusion = evaluation.confusionString();
        String output = evaluation.toString();

        // logging a few outputs for easy checking, and to prevent dead code elimination.
        logger.finer(confusion);
        logger.finer(output);

        // Test serialization
        Helpers.testModelSerialization(model,Event.class);

        // Test protobuf serialization
        Helpers.testModelProtoSerialization(model, Event.class, pair.getB());
    }

}

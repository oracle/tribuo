/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.fs;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.classification.Label;
import org.tribuo.util.Util;
import org.tribuo.util.infotheory.InformationTheory;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class mRMRTest {
    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(InformationTheory.class.getName());
        logger.setLevel(Level.WARNING);
        logger = Logger.getLogger(mRMR.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void parallelismTest() {
        Dataset<Label> dataset = MIMTest.createDataset();

        mRMR sequential = new mRMR(-1,10,1);
        SelectedFeatureSet sequentialSet = sequential.select(dataset);

        mRMR parallel = new mRMR(-1,10,4);
        SelectedFeatureSet parallelSet = parallel.select(dataset);

        assertEquals(sequentialSet.featureNames(),parallelSet.featureNames());
        double[] sequentialScores = Util.toPrimitiveDouble(sequentialSet.featureScores());
        double[] parallelScores = Util.toPrimitiveDouble(parallelSet.featureScores());
        assertArrayEquals(sequentialScores,parallelScores,1e-15);
    }
}

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
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.util.infotheory.InformationTheory;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MIMTest {

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(InformationTheory.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void mimTest() {
        LabelFactory lblFactory = new LabelFactory();
        MutableDataset<Label> dataset = new MutableDataset<>(new SimpleDataSourceProvenance("Test",lblFactory),lblFactory);

        String[] featureNames = new String[]{"A","B","C","D","E"};
        Label one = new Label("ONE");
        Label two = new Label("TWO");
        Example<Label> ex = new ArrayExample<>(one,featureNames,new double[]{0,0,0,0,0});
        dataset.add(ex);

        ex = new ArrayExample<>(one,featureNames,new double[]{1,0,1,0,0});
        dataset.add(ex);

        ex = new ArrayExample<>(one,featureNames,new double[]{0,0,0,1,0});
        dataset.add(ex);

        ex = new ArrayExample<>(one,featureNames,new double[]{1,1,0,0,0});
        dataset.add(ex);

        ex = new ArrayExample<>(two,featureNames,new double[]{0,0,1,0,1});
        dataset.add(ex);

        ex = new ArrayExample<>(two,featureNames,new double[]{1,1,1,0,1});
        dataset.add(ex);

        ex = new ArrayExample<>(two,featureNames,new double[]{0,1,1,1,1});
        dataset.add(ex);

        ex = new ArrayExample<>(two,featureNames,new double[]{1,1,0,1,1});
        dataset.add(ex);

        MIM mim = new MIM(2);

        SelectedFeatureSet sfs = mim.select(dataset);

        List<String> names = sfs.featureNames();
        List<Double> scores = sfs.featureScores();

        assertTrue(sfs.isOrdered());
        assertEquals(5,names.size());
        assertEquals(5,scores.size());

        assertEquals(Arrays.asList("E","B","C","D","A"),sfs.featureNames());
        assertEquals(Arrays.asList(1.0, 0.1887218755408671, 0.1887218755408671, 0.0487949406953985, 0.0),sfs.featureScores());
    }

}

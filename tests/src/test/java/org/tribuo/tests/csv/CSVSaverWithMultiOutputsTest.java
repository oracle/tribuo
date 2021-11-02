/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.tests.csv;

import org.tribuo.Dataset;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.csv.CSVSaver;
import org.tribuo.impl.ArrayExample;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.tests.Resources;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CSVSaverWithMultiOutputsTest {

    @Test
    public void savesMultiLabel() throws IOException {
        Set<String> responseNames = new HashSet<>(Arrays.asList("MONKEY", "PUZZLE", "TREE"));
        MultiLabelFactory factory = new MultiLabelFactory();

        MutableDataset<MultiLabel> before = new MutableDataset<>(null, factory);
        ArrayExample<MultiLabel> e = new ArrayExample<>(factory.generateOutput("MONKEY"));
        e.add(new Feature("A-MONKEY",1.0));
        e.add(new Feature("B-PUZZLE",0.0));
        e.add(new Feature("C-TREE",0.0));
        before.add(e);

        ArrayExample<MultiLabel> b = new ArrayExample<>(factory.generateOutput("MONKEY,TREE"));
        b.add(new Feature("A-MONKEY",1.0));
        b.add(new Feature("C-TREE",1.0));

        CSVSaver saver = new CSVSaver();
        File tmpFile = File.createTempFile("tribuo-csv-test","csv");
        tmpFile.deleteOnExit();
        Path tmp = tmpFile.toPath();
        saver.save(tmp, before, responseNames);
        // TODO use this to compare literal saver outputs
//        ByteArrayOutputStream baos = new ByteArrayOutputStream();
//        saver.save(baos, before, responseNames);
//        baos.flush();
//        System.out.println(new String(baos.toByteArray()));

        CSVLoader<MultiLabel> loader = new CSVLoader<>(factory);
        MutableDataset<MultiLabel> after = loader.load(tmp, responseNames);

        assertEquals(before.getData(), after.getData());
        assertEquals(before.getOutputIDInfo().size(), after.getOutputIDInfo().size());
        assertEquals(before.getFeatureIDMap().size(), after.getFeatureIDMap().size());
    }

    @Test
    public void loaderCanReconstructSavedMultiLabel() throws IOException {
        Path path = Resources.copyResourceToTmp("/org/tribuo/tests/csv/multilabel.csv");
        Set<String> responses = new HashSet<>(Arrays.asList("R1", "R2"));
        //
        // Load the csv
        CSVLoader<MultiLabel> loader = new CSVLoader<>(new MultiLabelFactory());
        MutableDataset<MultiLabel> before = loader.load(path, responses);
        //
        // Save the dataset
        File tmpFile = File.createTempFile("tribuo-csv-test","csv");
        tmpFile.deleteOnExit();
        Path tmp = tmpFile.toPath();
        new CSVSaver().save(tmp, before, responses);
        //
        // Reload and check that before & after are equivalent.
        MutableDataset<MultiLabel> after = loader.load(tmp, responses);
        //
        // TODO: better check for dataset equivalence?
        assertEquals(before.getData(), after.getData());
        assertEquals(before.getOutputIDInfo().size(), after.getOutputIDInfo().size());
        assertEquals(before.getFeatureIDMap().size(), after.getFeatureIDMap().size());
    }


    @Test
    public void savesMultipleRegression() throws IOException {
        String[] vars = new String[]{"dim1", "dim2"};
        Set<String> responseNames = new HashSet<>(Arrays.asList("dim1", "dim2"));
        RegressionFactory factory = new RegressionFactory();

        MutableDataset<Regressor> before = new MutableDataset<>(null, factory);
        ArrayExample<Regressor> e = new ArrayExample<>(new Regressor(vars, new double[]{0.1, 0.0}));
        e.add(new Feature("A", 1.0));
        e.add(new Feature("B", 0.0));
        e.add(new Feature("C", 0.0));
        before.add(e);

        ArrayExample<Regressor> b = new ArrayExample<>(new Regressor(vars, new double[]{0.0, 0.0}));
        b.add(new Feature("A", 1.0));
        b.add(new Feature("B", 0.0));
        b.add(new Feature("C", 0.1));
        before.add(b);

        CSVSaver saver = new CSVSaver();
        File tmpFile = File.createTempFile("tribuo-csv-test","csv");
        tmpFile.deleteOnExit();
        Path tmp = tmpFile.toPath();
        saver.save(tmp, before, responseNames);
        // TODO use this to compare literal saver outputs
//        ByteArrayOutputStream baos = new ByteArrayOutputStream();
//        saver.save(baos, before, responseNames);
//        baos.flush();
//        System.out.println(new String(baos.toByteArray()));

        CSVLoader<Regressor> loader = new CSVLoader<>(factory);
        MutableDataset<Regressor> after = loader.load(tmp, responseNames);

        assertEquals(before.getData(), after.getData());
        assertEquals(before.getOutputIDInfo().size(), after.getOutputIDInfo().size());
        assertEquals(before.getFeatureIDMap().size(), after.getFeatureIDMap().size());
    }


    @Test
    public void loaderCanReconstructSavedMultipleRegression() throws IOException {
        Path path = Resources.copyResourceToTmp("/org/tribuo/tests/csv/multioutput-regression.csv");
        Set<String> responses = new HashSet<>(Arrays.asList("R1", "R2"));
        CSVLoader<Regressor> loader = new CSVLoader<>(new RegressionFactory());
        Dataset<Regressor> before = loader.load(path, responses);

        File tmpFile = File.createTempFile("tribuo-csv-test","csv");
        tmpFile.deleteOnExit();
        Path tmp = tmpFile.toPath();
        new CSVSaver().save(tmp, before, responses);

        Dataset<Regressor> after = loader.load(tmp, responses);

        //
        // TODO: better check for dataset equivalence?
        assertEquals(before.getData(), after.getData());
        assertEquals(before.getOutputIDInfo().size(), after.getOutputIDInfo().size());
        assertEquals(before.getFeatureIDMap().size(), after.getFeatureIDMap().size());
    }


}
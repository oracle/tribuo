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

package org.tribuo.multilabel.example;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.time.OffsetDateTime;

/**
 * Generates three example train and test datasets, used for unit testing.
 * They don't necessarily have sensible classification boundaries,
 * it's for testing the machinery rather than accuracy.
 */
public class MultiLabelDataGenerator {

    private static final MultiLabelFactory factory = new MultiLabelFactory();

    private MultiLabelDataGenerator() {}

    /**
     * Simple training data for checking multi-label trainers.
     * @return Simple 3 class training data.
     */
    public static Dataset<MultiLabel> generateTrainData() {
        DataSourceProvenance provenance = new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), factory);
        MutableDataset<MultiLabel> dataset = new MutableDataset<>(provenance, factory);

        ArrayExample<MultiLabel> e = new ArrayExample<>(factory.generateOutput("MONKEY"));
        e.add(new Feature("A-MONKEY",1.0));
        e.add(new Feature("B-PUZZLE",0.0));
        e.add(new Feature("C-TREE",0.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("PUZZLE"));
        e.add(new Feature("A-MONKEY",0.0));
        e.add(new Feature("B-PUZZLE",1.0));
        e.add(new Feature("C-TREE",0.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("TREE"));
        e.add(new Feature("A-MONKEY",0.0));
        e.add(new Feature("B-PUZZLE",0.0));
        e.add(new Feature("C-TREE",1.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("MONKEY,TREE"));
        e.add(new Feature("A-MONKEY",1.0));
        e.add(new Feature("B-PUZZLE",0.0));
        e.add(new Feature("C-TREE",1.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("PUZZLE,MONKEY"));
        e.add(new Feature("A-MONKEY",1.0));
        e.add(new Feature("B-PUZZLE",1.0));
        e.add(new Feature("C-TREE",0.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("TREE,PUZZLE"));
        e.add(new Feature("A-MONKEY",0.0));
        e.add(new Feature("B-PUZZLE",1.0));
        e.add(new Feature("C-TREE",1.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("MONKEY,TREE,PUZZLE"));
        e.add(new Feature("A-MONKEY",0.5));
        e.add(new Feature("B-PUZZLE",0.5));
        e.add(new Feature("C-TREE",0.5));
        dataset.add(e);

        return dataset;
    }

    /**
     * Simple test data for checking multi-label trainers.
     * @return Simple 3 class test data.
     */
    public static Dataset<MultiLabel> generateTestData() {
        DataSourceProvenance provenance = new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), factory);
        MutableDataset<MultiLabel> dataset = new MutableDataset<>(provenance,factory);

        ArrayExample<MultiLabel> e = new ArrayExample<>(factory.generateOutput("MONKEY,PUZZLE,TREE"));
        e.add(new Feature("A-MONKEY",1.0));
        e.add(new Feature("B-PUZZLE",1.0));
        e.add(new Feature("C-TREE",1.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("MONKEY"));
        e.add(new Feature("A-MONKEY",1.0));
        e.add(new Feature("B-PUZZLE",0.0));
        e.add(new Feature("C-TREE",0.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("PUZZLE"));
        e.add(new Feature("A-MONKEY",0.0));
        e.add(new Feature("B-PUZZLE",1.0));
        e.add(new Feature("C-TREE",0.0));
        dataset.add(e);

        e = new ArrayExample<>(factory.generateOutput("TREE"));
        e.add(new Feature("A-MONKEY",0.0));
        e.add(new Feature("B-PUZZLE",0.0));
        e.add(new Feature("C-TREE",1.0));
        dataset.add(e);

        return dataset;
    }

    /**
     * Generate training and testing datasets.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<MultiLabel>,Dataset<MultiLabel>> generateDataset() {
        return new Pair<>(generateTrainData(),generateTestData());
    }

    /**
     * Generates an example with the feature ids 1,5,8, which does not intersect with the
     * ids used elsewhere in this class. This should make the example empty at prediction time.
     * @return An example with features {1:1.0,5:5.0,8:8.0}.
     */
    public static Example<MultiLabel> invalidSparseExample() {
        return new ArrayExample<>(new MultiLabel("MONKEY"),new String[]{"1","5","8"},new double[]{1.0,5.0,8.0});
    }

    /**
     * Generates an example with no features.
     * @return An example with no features.
     */
    public static Example<MultiLabel> emptyExample() {
        return new ArrayExample<>(new MultiLabel("MONKEY"),new String[]{},new double[]{});
    }

}

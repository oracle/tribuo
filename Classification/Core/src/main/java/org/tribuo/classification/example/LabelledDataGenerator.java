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

package org.tribuo.classification.example;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.time.OffsetDateTime;

/**
 * Generates three example train and test datasets, used for unit testing.
 * They don't necessarily have sensible classification boundaries,
 * it's for testing the machinery rather than accuracy.
 */
public final class LabelledDataGenerator {

    // final class with private constructor to ensure it's not instantiated.
    private LabelledDataGenerator() {}

    private static final LabelFactory labelFactory = new LabelFactory();

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}, and there are 4 classes,
     * {Foo,Bar,Baz,Quux}.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Label>,Dataset<Label>> denseTrainTest() {
        return denseTrainTest(-1.0);
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}, and there are 4 classes,
     * {Foo,Bar,Baz,Quux}.
     * @param negate Supply -1.0 to insert some negative values into the dataset.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Label>,Dataset<Label>> denseTrainTest(double negate) {
        DataSourceProvenance provenance = new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(),labelFactory);
        MutableDataset<Label> train = new MutableDataset<>(provenance,labelFactory);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Label("Foo"),names.clone(),values));
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Label("Foo"),names.clone(),values));

        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Label("Bar"),names.clone(),values));
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Label("Bar"),names.clone(),values));
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Label("Bar"),names.clone(),values));

        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Label("Baz"),names.clone(),values));
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Label("Baz"),names.clone(),values));
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Label("Baz"),names.clone(),values));

        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Label("Quux"),names.clone(),values));
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Label("Quux"),names.clone(),values));
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Label("Quux"),names.clone(),values));

        DataSourceProvenance testProvenance = new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(),labelFactory);
        MutableDataset<Label> test = new MutableDataset<>(testProvenance,labelFactory);

        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Label("Foo"),names.clone(),values));
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Label("Bar"),names.clone(),values));
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Label("Baz"),names.clone(),values));
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Label("Quux"),names.clone(),values));

        return new Pair<>(train,test);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data. It has the same
     * 4 classes {Foo,Bar,Baz,Quux}.
     * @return A pair of train and test datasets.
     */
    public static Pair<Dataset<Label>,Dataset<Label>> sparseTrainTest() {
        return sparseTrainTest(-1.0);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data. It has the same
     * 4 classes {Foo,Bar,Baz,Quux}.
     * @param negate Supply -1.0 to negate some values in this dataset.
     * @return A pair of train and test datasets.
     */
    public static Pair<Dataset<Label>,Dataset<Label>> sparseTrainTest(double negate) {
        DataSourceProvenance provenance = new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(),labelFactory);
        MutableDataset<Label> train = new MutableDataset<>(provenance,labelFactory);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"B","D","F","H"};
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"A","J","D","M"};
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));

        names = new String[]{"C","E","F","H"};
        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"E","G","F","I"};
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"J","K","C","E"};
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));

        names = new String[]{"E","A","K","J"};
        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Label("Baz"),names,values));
        names = new String[]{"B","C","E","H"};
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Label("Baz"),names,values));
        names = new String[]{"A","M","I","J"};
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Label("Baz"),names,values));

        names = new String[]{"Z","A","B","C"};
        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Label("Quux"),names,values));
        names = new String[]{"K","V","E","D"};
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Label("Quux"),names,values));
        names = new String[]{"B","G","E","A"};
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Label("Quux"),names,values));

        DataSourceProvenance testProvenance = new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(),labelFactory);
        MutableDataset<Label> test = new MutableDataset<>(testProvenance,labelFactory);

        names = new String[]{"AA","B","C","D"};
        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"B","BB","F","E"};
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"B","E","G","H"};
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Label("Baz"),names,values));
        names = new String[]{"B","CC","DD","EE"};
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Label("Quux"),names,values));

        return new Pair<>(train,test);
    }

    /**
     * Generates a pair of datasets with sparse features and unknown features
     * in the test data. Has binary labels {Foo,Bar}.
     * @return A pair of train and test datasets.
     */
    public static Pair<Dataset<Label>,Dataset<Label>> binarySparseTrainTest() {
        return binarySparseTrainTest(-1.0);
    }

    /**
     * Generates a pair of datasets with sparse features and unknown features
     * in the test data. Has binary labels {Foo,Bar}.
     * @param negate Supply -1.0 to negate some values in this dataset.
     * @return A pair of train and test datasets.
     */
    public static Pair<Dataset<Label>,Dataset<Label>> binarySparseTrainTest(double negate) {
        DataSourceProvenance provenance = new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(),labelFactory);
        MutableDataset<Label> train = new MutableDataset<>(provenance,labelFactory);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"B","D","F","H"};
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"A","J","D","M"};
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));

        names = new String[]{"C","E","F","H"};
        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"E","G","F","I"};
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"J","K","C","E"};
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));

        names = new String[]{"E","A","K","J"};
        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"B","C","E","H"};
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"A","M","I","J"};
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Label("Foo"),names,values));

        names = new String[]{"Z","A","B","C"};
        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"K","V","E","D"};
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"B","G","E","A"};
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Label("Bar"),names,values));

        DataSourceProvenance testProvenance = new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(),labelFactory);
        MutableDataset<Label> test = new MutableDataset<>(testProvenance,labelFactory);

        names = new String[]{"AA","B","C","D"};
        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"B","BB","F","E"};
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Label("Bar"),names,values));
        names = new String[]{"B","E","G","H"};
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Label("Foo"),names,values));
        names = new String[]{"B","CC","DD","EE"};
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Label("Bar"),names,values));

        return new Pair<>(train,test);
    }

    /**
     * Generates an example with the feature ids 1,5,8, which does not intersect with the
     * ids used elsewhere in this class. This should make the example empty at prediction time.
     * @return An example with features {1:1.0,5:5.0,8:8.0}.
     */
    public static Example<Label> invalidSparseExample() {
        return new ArrayExample<>(new Label("Foo"),new String[]{"1","5","8"},new double[]{1.0,5.0,8.0});
    }

    /**
     * Generates an example with no features.
     * @return An example with no features.
     */
    public static Example<Label> emptyExample() {
        return new ArrayExample<>(new Label("Foo"),new String[]{},new double[]{});
    }

}

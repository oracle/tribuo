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

package org.tribuo.regression.example;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;

import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Generates two example train and test datasets, used for unit testing.
 * They don't necessarily have linear regressed values,
 * it's for testing the machinery rather than accuracy.
 * <p>
 * Also can generate a variety of single dimensional gaussian datasets.
 */
public abstract class RegressionDataGenerator {

    /**
     * Name of the first output dimension.
     */
    public static final String firstDimensionName = "dim1";
    /**
     * Name of the second output dimension.
     */
    public static final String secondDimensionName= "dim2";
    /**
     * Name of the third output dimension.
     */
    public static final String thirdDimensionName = "dim3";

    /**
     * Name of the single dimension.
     */
    public static final String SINGLE_DIM_NAME = "REGRESSOR";

    private static final RegressionFactory REGRESSION_FACTORY = new RegressionFactory();
    private static final String[] dimensionNames = new String[]{firstDimensionName,secondDimensionName};
    private static final String[] threeNames = new String[]{firstDimensionName,secondDimensionName,thirdDimensionName};

    /**
     * Abstract utility class with private constructor.
     */
    private RegressionDataGenerator() {}

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> multiDimDenseTrainTest() {
        return multiDimDenseTrainTest(-1.0);
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}.
     * @param negate Supply -1.0 to negate some features.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> multiDimDenseTrainTest(double negate) {
        MutableDataset<Regressor> train = new MutableDataset<>(new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), REGRESSION_FACTORY), REGRESSION_FACTORY);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{5.0,-5.0}),names,values));
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{5.8,-5.8}),names.clone(),values));
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{8.0,-8.0}),names.clone(),values));

        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names.clone(),values));
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names.clone(),values));
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names.clone(),values));

        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names.clone(),values));
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names.clone(),values));
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names.clone(),values));

        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names.clone(),values));
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names.clone(),values));
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names.clone(),values));

        MutableDataset<Regressor> test = new MutableDataset<>(new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), REGRESSION_FACTORY), REGRESSION_FACTORY);

        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{5.1,-5.1}),names.clone(),values));
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names.clone(),values));
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names.clone(),values));
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names.clone(),values));

        return new Pair<>(train,test);
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}.
     * @param negate Supply -1.0 to negate some features.
     * @param remapIndices If true invert the indices of the output features.
     *                     Warning: this should only be used as part of unit testing, it is not expected from
     *                     standard datasets.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> threeDimDenseTrainTest(double negate, boolean remapIndices) {
        MutableDataset<Regressor> train = new MutableDataset<>(new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), REGRESSION_FACTORY), REGRESSION_FACTORY);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{5.0,-5.0,0.0}),names,values));
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{5.8,-5.8,1.0}),names,values));
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{8.0,-8.0,9.0}),names,values));

        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{10.0,-10.0,0.5}),names,values));
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{10.0,-10.0,0.5}),names,values));
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{10.0,-10.0,0.5}),names,values));

        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{20,-20,5.0}),names,values));
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{20,-20,4.0}),names,values));
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{20,-20,2.0}),names,values));

        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{50,-50,10}),names,values));
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{50,-50,15}),names,values));
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Regressor(threeNames,new double[]{50,-50,10}),names,values));

        MutableDataset<Regressor> test = new MutableDataset<>(new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), REGRESSION_FACTORY), REGRESSION_FACTORY);

        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Regressor(threeNames,new double[]{5.1,-5.1,1.2}),names,values));
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Regressor(threeNames,new double[]{10.0,-10.0,0.5}),names,values));
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Regressor(threeNames,new double[]{20,-20,6.0}),names,values));
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Regressor(threeNames,new double[]{50,-50,10}),names,values));

        if (remapIndices) {
            Map<Regressor,Integer> mapping = new HashMap<>();
            mapping.put(new Regressor.DimensionTuple(firstDimensionName,Double.NaN),2);
            mapping.put(new Regressor.DimensionTuple(secondDimensionName,Double.NaN),0);
            mapping.put(new Regressor.DimensionTuple(thirdDimensionName,Double.NaN),1);
            ImmutableOutputInfo<Regressor> newInfo = REGRESSION_FACTORY.constructInfoForExternalModel(mapping);

            ImmutableDataset<Regressor> newTrain = ImmutableDataset.copyDataset(train, train.getFeatureIDMap(), newInfo);
            ImmutableDataset<Regressor> newTest = ImmutableDataset.copyDataset(test, train.getFeatureIDMap(), newInfo);

            return new Pair<>(newTrain, newTest);
        } else {
            return new Pair<>(train, test);
        }
    }

    /** 
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> multiDimSparseTrainTest() {
        return multiDimSparseTrainTest(-1.0);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data.
     * @param negate Supply -1.0 to negate some features.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> multiDimSparseTrainTest(double negate) {
        MutableDataset<Regressor> train = new MutableDataset<>(new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), REGRESSION_FACTORY), REGRESSION_FACTORY);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{5.0,-5.0}),names,values));
        names = new String[]{"B","D","F","H"};
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{5.8,-5.8}),names,values));
        names = new String[]{"A","J","D","M"};
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{8.0,-8.0}),names,values));

        names = new String[]{"C","E","F","H"};
        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names,values));
        names = new String[]{"E","G","F","I"};
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names,values));
        names = new String[]{"J","K","C","E"};
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names,values));

        names = new String[]{"E","A","K","J"};
        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names,values));
        names = new String[]{"B","C","E","H"};
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names,values));
        names = new String[]{"A","M","I","J"};
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names,values));

        names = new String[]{"Z","A","B","C"};
        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names,values));
        names = new String[]{"K","V","E","D"};
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names,values));
        names = new String[]{"B","G","E","A"};
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names,values));

        MutableDataset<Regressor> test = new MutableDataset<>(new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), REGRESSION_FACTORY), REGRESSION_FACTORY);

        names = new String[]{"AA","B","C","D"};
        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{5.5,-5.5}),names,values));
        names = new String[]{"B","BB","F","E"};
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{10.0,-10.0}),names,values));
        names = new String[]{"B","E","G","H"};
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{20,-20}),names,values));
        names = new String[]{"B","CC","DD","EE"};
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Regressor(dimensionNames,new double[]{50,-50}),names,values));

        return new Pair<>(train,test);
    }

    /**
     * Generates an example with the feature ids 1,5,8, which does not intersect with the
     * ids used elsewhere in this class. This should make the example empty at prediction time.
     * @return An example with features {1:1.0,5:5.0,8:8.0}.
     */
    public static Example<Regressor> invalidMultiDimSparseExample() {
        return new ArrayExample<>(new Regressor(dimensionNames,new double[]{1,-1}),new String[]{"1","5","8"},new double[]{1.0,5.0,8.0});
    }

    /**
     * Generates an example with no features.
     * @return An example with no features.
     */
    public static Example<Regressor> emptyMultiDimExample() {
        return new ArrayExample<>(new Regressor(dimensionNames,new double[]{1,-1}),new String[]{},new double[]{});
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> denseTrainTest() {
        return denseTrainTest(-1.0);
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}.
     * @param negate Supply -1.0 to negate some values in this dataset.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> denseTrainTest(double negate) {
        DataSourceProvenance provenance = new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), REGRESSION_FACTORY);
        MutableDataset<Regressor> train = new MutableDataset<>(provenance, REGRESSION_FACTORY);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,5.0),names,values));
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,5.8),names,values));
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,8.0),names,values));

        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));

        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));

        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));

        DataSourceProvenance testProvenance = new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), REGRESSION_FACTORY);
        MutableDataset<Regressor> test = new MutableDataset<>(testProvenance, REGRESSION_FACTORY);

        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,5.1),names,values));
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));

        return new Pair<>(train,test);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> sparseTrainTest() {
        return sparseTrainTest(-1.0);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data.
     * @param negate Supply -1.0 to negate some values in this dataset.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Regressor>,Dataset<Regressor>> sparseTrainTest(double negate) {
        DataSourceProvenance provenance = new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), REGRESSION_FACTORY);
        MutableDataset<Regressor> train = new MutableDataset<>(provenance, REGRESSION_FACTORY);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,5.0),names,values));
        names = new String[]{"B","D","F","H"};
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,5.8),names,values));
        names = new String[]{"A","J","D","M"};
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,8.0),names,values));

        names = new String[]{"C","E","F","H"};
        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));
        names = new String[]{"E","G","F","I"};
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));
        names = new String[]{"J","K","C","E"};
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));

        names = new String[]{"E","A","K","J"};
        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));
        names = new String[]{"B","C","E","H"};
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));
        names = new String[]{"A","M","I","J"};
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));

        names = new String[]{"Z","A","B","C"};
        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));
        names = new String[]{"K","V","E","D"};
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));
        names = new String[]{"B","G","E","A"};
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));

        DataSourceProvenance testProvenance = new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), REGRESSION_FACTORY);
        MutableDataset<Regressor> test = new MutableDataset<>(testProvenance, REGRESSION_FACTORY);

        names = new String[]{"AA","B","C","D"};
        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,5.5),names,values));
        names = new String[]{"B","BB","F","E"};
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,10.0),names,values));
        names = new String[]{"B","E","G","H"};
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,20),names,values));
        names = new String[]{"B","CC","DD","EE"};
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,50),names,values));

        return new Pair<>(train,test);
    }

    /**
     * Generates an example with the feature ids 1,5,8, which does not intersect with the
     * ids used elsewhere in this class. This should make the example empty at prediction time.
     * @return An example with features {1:1.0,5:5.0,8:8.0}.
     */
    public static Example<Regressor> invalidSparseExample() {
        return new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,1),new String[]{"1","5","8"},new double[]{1.0,5.0,8.0});
    }

    /**
     * Generates an example with no features.
     * @return An example with no features.
     */
    public static Example<Regressor> emptyExample() {
        return new ArrayExample<>(new Regressor(SINGLE_DIM_NAME,1),new String[]{},new double[]{});
    }

}

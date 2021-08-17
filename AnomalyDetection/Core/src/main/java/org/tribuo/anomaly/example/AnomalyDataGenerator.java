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

package org.tribuo.anomaly.example;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.anomaly.AnomalyFactory;
import org.tribuo.anomaly.Event;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.tribuo.anomaly.AnomalyFactory.ANOMALOUS_EVENT;
import static org.tribuo.anomaly.AnomalyFactory.EXPECTED_EVENT;

/**
 * Generates three example train and test datasets, used for unit testing.
 * They don't necessarily have sensible boundaries,
 * it's for testing the machinery rather than accuracy.
 * <p>
 * Also has a dataset generator which returns a training dataset
 * with no anomalies sampled from a single gaussian, and a test dataset
 * sampled from two gaussians where the second is labelled anomalous.
 * <p>
 * For most use cases that are not unit tests, it is recommended to use
 * {@link GaussianAnomalyDataSource} which has similar functionality but
 * is more flexible and configurable.
 */
public abstract class AnomalyDataGenerator {

    private static AnomalyFactory anomalyFactory = new AnomalyFactory();

    /**
     * Generates two datasets, one without anomalies drawn from a single gaussian
     * and the second drawn from a mixture of two gaussians, with the second tagged
     * anomalous.
     *
     * Generates 200 training examples and 200 test examples, with 20% anomalies.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Event>,Dataset<Event>> gaussianAnomaly() {
        return gaussianAnomaly(200,0.2);
    }

    /**
     * Generates two datasets, one without anomalies drawn from a single gaussian
     * and the second drawn from a mixture of two gaussians, with the second tagged
     * anomalous.
     *
     * @param size The number of points to sample for each dataset.
     * @param fractionAnomalous The fraction of anomalous data to generate.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Event>,Dataset<Event>> gaussianAnomaly(long size, double fractionAnomalous) {
        if (size < 1) {
            throw new IllegalArgumentException("Size must be a positive number, received " + size);
        } else if ((fractionAnomalous > 1) || (fractionAnomalous < 0)) {
            throw new IllegalArgumentException("FractionAnomalous must be between zero and one, received " + fractionAnomalous);
        }
        Random rng = new Random(1L);

        // Dataset parameters
        String[] featureNames = new String[]{"A","B","C","D","E"};
        double[] expectedMeans = new double[]{1.0,2.0,1.0,2.0,5.0};
        double[] anomalousMeans = new double[]{-2.0,2.0,-2.0,2.0,-10.0};
        double[] variances = new double[]{1.0,0.5,0.25,1.0,0.1};

        List<Example<Event>> trainingData = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            List<Feature> featureList = generateFeatures(rng,featureNames,expectedMeans,variances);
            trainingData.add(new ArrayExample<>(EXPECTED_EVENT,featureList));
        }

        List<Example<Event>> testingData = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            double draw = rng.nextDouble();
            if (draw < fractionAnomalous) {
                List<Feature> featureList = generateFeatures(rng, featureNames, anomalousMeans, variances);
                testingData.add(new ArrayExample<>(ANOMALOUS_EVENT, featureList));
            } else {
                List<Feature> featureList = generateFeatures(rng, featureNames, expectedMeans, variances);
                testingData.add(new ArrayExample<>(EXPECTED_EVENT, featureList));
            }
        }

        SimpleDataSourceProvenance trainingProvenance = new SimpleDataSourceProvenance("Anomaly training data",anomalyFactory);
        MutableDataset<Event> train = new MutableDataset<>(new ListDataSource<>(trainingData,anomalyFactory,trainingProvenance));
        SimpleDataSourceProvenance testingProvenance = new SimpleDataSourceProvenance("Anomaly testing data",anomalyFactory);
        MutableDataset<Event> test = new MutableDataset<>(new ListDataSource<>(testingData,anomalyFactory,testingProvenance));

        return new Pair<>(train,test);
    }

    private static List<Feature> generateFeatures(Random rng, String[] names, double[] means, double[] variances) {
        if ((names.length != means.length) || (names.length != variances.length)) {
            throw new IllegalArgumentException("Names, means and variances must be the same length");
        }

        List<Feature> features = new ArrayList<>();

        for (int i = 0; i < names.length; i++) {
            double value = (rng.nextGaussian() * Math.sqrt(variances[i])) + means[i];
            features.add(new Feature(names[i],value));
        }

        return features;
    }

    /**
     * Makes a simple dataset for training and testing.
     * <p>
     * Used for smoke testing, doesn't have a real boundary.
     * @return A pair containing a training dataset and a testing dataset.
     */
    public static Pair<Dataset<Event>,Dataset<Event>> denseTrainTest() {
        return denseTrainTest(-1.0);
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}, and there are 4 clusters,
     * {0,1,2,3}.
     * @param negate Supply -1.0 to negate some feature values.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Event>,Dataset<Event>> denseTrainTest(double negate) {
        MutableDataset<Event> train = new MutableDataset<>(new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), anomalyFactory), anomalyFactory);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        MutableDataset<Event> test = new MutableDataset<>(new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), anomalyFactory), anomalyFactory);

        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        return new Pair<>(train,test);
    }

    /**
     * Makes a simple dataset for training and testing.
     * <p>
     * Used for smoke testing, doesn't have a real boundary.
     * @return A pair containing a training dataset and a testing dataset.
     */
    public static Pair<Dataset<Event>,Dataset<Event>> sparseTrainTest() {
        return sparseTrainTest(-1.0);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data. It has the same
     * 4 clusters {0,1,2,3}.
     * @param negate Supply -1.0 to negate some feature values.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<Event>,Dataset<Event>> sparseTrainTest(double negate) {
        MutableDataset<Event> train = new MutableDataset<>(new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), anomalyFactory), anomalyFactory);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"B","D","F","H"};
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"A","J","D","M"};
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        names = new String[]{"C","E","F","H"};
        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"E","G","F","I"};
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"J","K","C","E"};
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        names = new String[]{"E","A","K","J"};
        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"B","C","E","H"};
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"A","M","I","J"};
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        names = new String[]{"Z","A","B","C"};
        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"K","V","E","D"};
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"B","G","E","A"};
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        MutableDataset<Event> test = new MutableDataset<>(new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(), anomalyFactory), anomalyFactory);

        names = new String[]{"AA","B","C","D"};
        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"B","BB","F","E"};
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"B","E","G","H"};
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));
        names = new String[]{"B","CC","DD","EE"};
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(EXPECTED_EVENT,names,values));

        return new Pair<>(train,test);
    }

    /**
     * Generates an example with the feature ids 1,5,8, which does not intersect with the
     * ids used elsewhere in this class. This should make the example empty at prediction time.
     * @return An example with features {1:1.0,5:5.0,8:8.0}.
     */
    public static Example<Event> invalidSparseExample() {
        return new ArrayExample<>(EXPECTED_EVENT,new String[]{"1","5","8"},new double[]{1.0,5.0,8.0});
    }

    /**
     * Generates an example with no features.
     * @return An example with no features.
     */
    public static Example<Event> emptyExample() {
        return new ArrayExample<>(EXPECTED_EVENT,new String[]{},new double[]{});
    }
}

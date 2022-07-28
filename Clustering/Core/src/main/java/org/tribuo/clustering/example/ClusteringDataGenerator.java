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

package org.tribuo.clustering.example;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.impl.ArrayExample;
import org.tribuo.math.distributions.MultivariateNormalDistribution;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates three example train and test datasets, used for unit testing.
 * They don't necessarily have sensible cluster boundaries,
 * it's for testing the machinery rather than accuracy.
 * <p>
 * Also has a dataset generator which returns a dataset
 * sampled from a mixture of 2 dimensional gaussians.
 */
public abstract class ClusteringDataGenerator {

    private static ClusteringFactory clusteringFactory = new ClusteringFactory();

    /**
     * Generates a dataset drawn from a mixture of 5 2d gaussians.
     *
     * @param size The number of points to sample for the dataset.
     * @param seed The RNG seed.
     * @return A pair of datasets.
     */
    public static Dataset<ClusterID> gaussianClusters(long size, long seed) {
        if (size < 1) {
            throw new IllegalArgumentException("Size must be a positive number, received " + size);
        }
        Random rng = new Random(seed);

        // Dataset parameters
        String[] featureNames = new String[]{"A","B"};
        double[] mixingPMF = new double[]{0.1,0.35,0.05,0.25,0.25};
        double[] mixingCDF = Util.generateCDF(mixingPMF);
        MultivariateNormalDistribution first = new MultivariateNormalDistribution(
                new double[]{0.0,0.0}, new double[][]{{1.0,0.0},{0.0,1.0}}, rng.nextInt(), true
        );
        MultivariateNormalDistribution second = new MultivariateNormalDistribution(
                new double[]{5.0,5.0}, new double[][]{{1.0,0.0},{0.0,1.0}}, rng.nextInt(), true
        );
        MultivariateNormalDistribution third = new MultivariateNormalDistribution(
                new double[]{2.5,2.5}, new double[][]{{1.0,0.5},{0.5,1.0}}, rng.nextInt(), true
        );
        MultivariateNormalDistribution fourth = new MultivariateNormalDistribution(
                new double[]{10.0,0.0}, new double[][]{{0.1,0.0},{0.0,0.1}}, rng.nextInt(), true
        );
        MultivariateNormalDistribution fifth = new MultivariateNormalDistribution(
                new double[]{-1.0,0.0}, new double[][]{{1.0,0.0},{0.0,0.1}}, rng.nextInt(), true
        );
        MultivariateNormalDistribution[] gaussians = new MultivariateNormalDistribution[]{first,second,third,fourth,fifth};

        List<Example<ClusterID>> trainingData = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            int centroid = Util.sampleFromCDF(mixingCDF,rng);
            double[] sample = gaussians[centroid].sampleArray();
            trainingData.add(new ArrayExample<>(new ClusterID(centroid),featureNames,sample));
        }

        SimpleDataSourceProvenance trainingProvenance = new SimpleDataSourceProvenance("Generated clustering data",clusteringFactory);
        return new MutableDataset<>(new ListDataSource<>(trainingData,clusteringFactory,trainingProvenance));
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}, and there are 4 clusters,
     * {0,1,2,3}.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<ClusterID>,Dataset<ClusterID>> denseTrainTest() {
        return denseTrainTest(-1.0);
    }

    /**
     * Generates a train/test dataset pair which is dense in the features,
     * each example has 4 features,{A,B,C,D}, and there are 4 clusters,
     * {0,1,2,3}.
     * @param negate Supply -1.0 to negate some feature values.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<ClusterID>,Dataset<ClusterID>> denseTrainTest(double negate) {
        MutableDataset<ClusterID> train = new MutableDataset<>(new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), clusteringFactory), clusteringFactory);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new ClusterID(1),names,values));
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new ClusterID(1),names.clone(),values));
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new ClusterID(1),names.clone(),values));

        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new ClusterID(2),names.clone(),values));
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new ClusterID(2),names.clone(),values));
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new ClusterID(2),names.clone(),values));

        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new ClusterID(3),names.clone(),values));
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new ClusterID(3),names.clone(),values));
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new ClusterID(3),names.clone(),values));

        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new ClusterID(0),names.clone(),values));
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new ClusterID(0),names.clone(),values));
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new ClusterID(0),names.clone(),values));

        MutableDataset<ClusterID> test = new MutableDataset<>(new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(),clusteringFactory), clusteringFactory);

        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new ClusterID(1),names.clone(),values));
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new ClusterID(2),names.clone(),values));
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new ClusterID(3),names.clone(),values));
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new ClusterID(0),names.clone(),values));

        return new Pair<>(train,test);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data. It has the same
     * 4 clusters {0,1,2,3}.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<ClusterID>,Dataset<ClusterID>> sparseTrainTest() {
        return sparseTrainTest(-1.0);
    }

    /**
     * Generates a pair of datasets, where the features are sparse,
     * and unknown features appear in the test data. It has the same
     * 4 clusters {0,1,2,3}.
     * @param negate Supply -1.0 to negate some feature values.
     * @return A pair of datasets.
     */
    public static Pair<Dataset<ClusterID>,Dataset<ClusterID>> sparseTrainTest(double negate) {
        MutableDataset<ClusterID> train = new MutableDataset<>(new SimpleDataSourceProvenance("TrainingData", OffsetDateTime.now(), clusteringFactory), clusteringFactory);

        String[] names = new String[]{"A","B","C","D"};
        double[] values = new double[]{1.0,0.5,1.0,negate*1.0};
        train.add(new ArrayExample<>(new ClusterID(1),names,values));
        names = new String[]{"B","D","F","H"};
        values = new double[]{1.5,0.35,1.3,negate*1.2};
        train.add(new ArrayExample<>(new ClusterID(1),names,values));
        names = new String[]{"A","J","D","M"};
        values = new double[]{1.2,0.45,1.5,negate*1.0};
        train.add(new ArrayExample<>(new ClusterID(1),names,values));

        names = new String[]{"C","E","F","H"};
        values = new double[]{negate*1.1,0.55,negate*1.5,0.5};
        train.add(new ArrayExample<>(new ClusterID(2),names,values));
        names = new String[]{"E","G","F","I"};
        values = new double[]{negate*1.5,0.25,negate*1,0.125};
        train.add(new ArrayExample<>(new ClusterID(2),names,values));
        names = new String[]{"J","K","C","E"};
        values = new double[]{negate*1,0.5,negate*1.123,0.123};
        train.add(new ArrayExample<>(new ClusterID(2),names,values));

        names = new String[]{"E","A","K","J"};
        values = new double[]{1.5,5.0,0.5,4.5};
        train.add(new ArrayExample<>(new ClusterID(3),names,values));
        names = new String[]{"B","C","E","H"};
        values = new double[]{1.234,5.1235,0.1235,6.0};
        train.add(new ArrayExample<>(new ClusterID(3),names,values));
        names = new String[]{"A","M","I","J"};
        values = new double[]{1.734,4.5,0.5123,5.5};
        train.add(new ArrayExample<>(new ClusterID(3),names,values));

        names = new String[]{"Z","A","B","C"};
        values = new double[]{negate*1,0.25,5,10.0};
        train.add(new ArrayExample<>(new ClusterID(0),names,values));
        names = new String[]{"K","V","E","D"};
        values = new double[]{negate*1.4,0.55,5.65,12.0};
        train.add(new ArrayExample<>(new ClusterID(0),names,values));
        names = new String[]{"B","G","E","A"};
        values = new double[]{negate*1.9,0.25,5.9,15};
        train.add(new ArrayExample<>(new ClusterID(0),names,values));

        MutableDataset<ClusterID> test = new MutableDataset<>(new SimpleDataSourceProvenance("TestingData", OffsetDateTime.now(),clusteringFactory), clusteringFactory);

        names = new String[]{"AA","B","C","D"};
        values = new double[]{2.0,0.45,3.5,negate*2.0};
        test.add(new ArrayExample<>(new ClusterID(1),names,values));
        names = new String[]{"B","BB","F","E"};
        values = new double[]{negate*2.0,0.55,negate*2.5,2.5};
        test.add(new ArrayExample<>(new ClusterID(2),names,values));
        names = new String[]{"B","E","G","H"};
        values = new double[]{1.75,5.0,1.0,6.5};
        test.add(new ArrayExample<>(new ClusterID(3),names,values));
        names = new String[]{"B","CC","DD","EE"};
        values = new double[]{negate*1.5,0.25,5.0,20.0};
        test.add(new ArrayExample<>(new ClusterID(0),names,values));

        return new Pair<>(train,test);
    }

    /**
     * Generates an example with the feature ids 1,5,8, which does not intersect with the
     * ids used elsewhere in this class. This should make the example empty at prediction time.
     * @return An example with features {1:1.0,5:5.0,8:8.0}.
     */
    public static Example<ClusterID> invalidSparseExample() {
        return new ArrayExample<>(new ClusterID(1),new String[]{"1","5","8"},new double[]{1.0,5.0,8.0});
    }

    /**
     * Generates an example with no features.
     * @return An example with no features.
     */
    public static Example<ClusterID> emptyExample() {
        return new ArrayExample<>(new ClusterID(1),new String[]{},new double[]{});
    }
}

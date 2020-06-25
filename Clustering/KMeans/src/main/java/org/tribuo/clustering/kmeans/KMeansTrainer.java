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
package org.tribuo.clustering.kmeans;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.StreamUtil;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ImmutableClusteringInfo;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A K-Means trainer, which generates a K-means clustering of the supplied
 * data. The model finds the centres, and then predict needs to be
 * called to infer the centre assignments for the input data.
 * <p>
 * It's slightly contorted to fit the Tribuo Trainer and Model API, as the cluster assignments
 * can only be retrieved from the model after training, and require re-evaluating each example.
 * <p>
 * The Trainer has a parameterised distance function, and a selectable number
 * of threads used in the training step. The thread pool is local to an invocation of train,
 * so there can be multiple concurrent trainings.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public class KMeansTrainer implements Trainer<ClusterID> {
    private static final Logger logger = Logger.getLogger(KMeansTrainer.class.getName());

    /**
     * Possible distance functions.
     */
    public enum Distance {
        /**
         * Euclidean (or l2) distance.
         */
        EUCLIDEAN,
        /**
         * Cosine similarity as a distance measure.
         */
        COSINE,
        /**
         * L1 (or Manhattan) distance.
         */
        L1
    }

    @Config(mandatory = true,description="Number of centroids (i.e. the \"k\" in k-means).")
    private int centroids;

    @Config(mandatory = true,description="The number of iterations to run.")
    private int iterations;

    @Config(mandatory = true,description="The distance function to use.")
    private Distance distanceType;

    @Config(description="The number of threads to use for training.")
    private int numThreads = 1;

    @Config(mandatory = true,description="The seed to use for the RNG.")
    private long seed;

    private SplittableRandom rng;

    private int trainInvocationCounter;

    /**
     * for olcut.
     */
    private KMeansTrainer() {}

    /**
     * Constructs a K-Means trainer using the supplied parameters.
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param distanceType The distance function.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public KMeansTrainer(int centroids, int iterations, Distance distanceType, int numThreads, long seed) {
        this.centroids = centroids;
        this.iterations = iterations;
        this.distanceType = distanceType;
        this.numThreads = numThreads;
        this.seed = seed;
        postConfig();
    }

    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);
    }

    @Override
    public KMeansModel train(Dataset<ClusterID> examples, Map<String, Provenance> runProvenance) {
        // Creates a new local RNG and adds one to the invocation count.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        synchronized(this) {
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        DenseVector[] centroidVectors = initialiseCentroids(centroids,examples,featureMap,localRNG);

        ForkJoinPool fjp = new ForkJoinPool(numThreads);

        int[] oldCentre = new int[examples.size()];
        SparseVector[] data = new SparseVector[examples.size()];
        double[] weights = new double[examples.size()];
        int n = 0;
        for (Example<ClusterID> example : examples) {
            weights[n] = example.getWeight();
            data[n] = SparseVector.createSparseVector(example,featureMap,false);
            oldCentre[n] = -1;
            n++;
        }

        Map<Integer,List<Integer>> clusterAssignments = new HashMap<>();
        for (int i = 0; i < centroids; i++) {
            clusterAssignments.put(i,Collections.synchronizedList(new ArrayList<>()));
        }

        boolean converged = false;

        for (int i = 0; (i < iterations) && !converged; i++) {
            //logger.log(Level.INFO,"Beginning iteration " + i);
            AtomicInteger changeCounter = new AtomicInteger(0);

            for (Entry<Integer,List<Integer>> e : clusterAssignments.entrySet()) {
                e.getValue().clear();
            }

            // E step
            Stream<SparseVector> vecStream = Arrays.stream(data);
            Stream<Integer> intStream = IntStream.range(0,data.length).boxed();
            Stream<IntAndVector> eStream;
            if (numThreads > 1) {
                eStream = StreamUtil.boundParallelism(StreamUtil.zip(intStream,vecStream,IntAndVector::new).parallel());
            } else {
                eStream = StreamUtil.zip(intStream,vecStream,IntAndVector::new);
            }
            try {
                fjp.submit(() -> eStream.forEach((IntAndVector e) -> {
                    double minDist = Double.POSITIVE_INFINITY;
                    int clusterID = -1;
                    int id = e.idx;
                    SparseVector vector = e.vector;
                    for (int j = 0; j < centroids; j++) {
                        DenseVector cluster = centroidVectors[j];
                        double distance;
                        switch (distanceType) {
                            case EUCLIDEAN:
                                distance = cluster.euclideanDistance(vector);
                                break;
                            case COSINE:
                                distance = cluster.cosineDistance(vector);
                                break;
                            case L1:
                                distance = cluster.l1Distance(vector);
                                break;
                            default:
                                throw new IllegalStateException("Unknown distance " + distanceType);
                        }
                        if (distance < minDist) {
                            minDist = distance;
                            clusterID = j;
                        }
                    }

                    clusterAssignments.get(clusterID).add(id);
                    if (oldCentre[id] != clusterID) {
                        // Changed the centroid of this vector.
                        oldCentre[id] = clusterID;
                        changeCounter.incrementAndGet();
                    }
                })).get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Parallel execution failed",e);
            }
            //logger.log(Level.INFO, "E step completed. " + changeCounter.get() + " words updated.");

            mStep(fjp,centroidVectors,clusterAssignments,data,weights);

            logger.log(Level.INFO, "Iteration " + i + " completed. " + changeCounter.get() + " examples updated.");

            if (changeCounter.get() == 0) {
                converged = true;
                logger.log(Level.INFO, "K-Means converged at iteration " + i);
            }
        }


        Map<Integer,MutableLong> counts = new HashMap<>();
        for (Entry<Integer,List<Integer>> e : clusterAssignments.entrySet()) {
            counts.put(e.getKey(),new MutableLong(e.getValue().size()));
        }

        ImmutableOutputInfo<ClusterID> outputMap = new ImmutableClusteringInfo(counts);

        ModelProvenance provenance = new ModelProvenance(KMeansModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);

        return new KMeansModel("",provenance,featureMap,outputMap,centroidVectors, distanceType);
    }

    @Override
    public KMeansModel train(Dataset<ClusterID> dataset) {
        return train(dataset,Collections.emptyMap());
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    /**
     * Initialisation method called at the start of each train call.
     *
     * Used to allow overriding for kmeans++, kmedoids etc.
     *
     * @param centroids The number of centroids to create.
     * @param examples The dataset to use.
     * @param featureMap The feature map to use for centroid sampling.
     * @param rng The RNG to use.
     * @return A {@link DenseVector} array of centroids.
     */
    protected static DenseVector[] initialiseCentroids(int centroids, Dataset<ClusterID> examples, ImmutableFeatureMap featureMap, SplittableRandom rng) {
        DenseVector[] centroidVectors = new DenseVector[centroids];
        int numFeatures = featureMap.size();
        for (int i = 0; i < centroids; i++) {
            double[] newCentroid = new double[numFeatures];

            for (int j = 0; j < numFeatures; j++) {
                newCentroid[j] = featureMap.get(j).uniformSample(rng);
            }

            centroidVectors[i] = DenseVector.createDenseVector(newCentroid);
        }
        return centroidVectors;
    }

    protected void mStep(ForkJoinPool fjp, DenseVector[] centroidVectors, Map<Integer,List<Integer>> clusterAssignments, SparseVector[] data, double[] weights) {
        // M step
        Stream<Entry<Integer,List<Integer>>> mStream;
        if (numThreads > 1) {
            mStream = StreamUtil.boundParallelism(clusterAssignments.entrySet().stream().parallel());
        } else {
            mStream = clusterAssignments.entrySet().stream();
        }
        try {
            fjp.submit(() -> mStream.forEach((e) -> {
                DenseVector newCentroid = centroidVectors[e.getKey()];
                newCentroid.fill(0.0);

                int counter = 0;
                for (Integer idx : e.getValue()) {
                    newCentroid.intersectAndAddInPlace(data[idx],(double f) -> f * weights[idx]);
                    counter++;
                }
                if (counter > 0) {
                    newCentroid.scaleInPlace(1.0/counter);
                }
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Parallel execution failed",e);
        }
    }

    @Override
    public String toString() {
        return "KMeansTrainer(centroids="+centroids+",distanceType="+ distanceType +",seed="+seed+",numThreads="+numThreads+")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    /**
     * Tuple of index and position. One day it'll be a record, but not today.
     */
    static class IntAndVector {
        final int idx;
        final SparseVector vector;

        public IntAndVector(int idx, SparseVector vector) {
            this.idx = idx;
            this.vector = vector;
        }
    }
}
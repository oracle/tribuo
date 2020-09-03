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
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.util.Util;

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
 * <p>
 * For more on optional kmeans++ initialisation, see:
 * <pre>
 * D. Arthur, S. Vassilvitskii.
 * "K-Means++: The Advantages of Careful Seeding"
 * <a href="https://theory.stanford.edu/~sergei/papers/kMeansPP-soda">PDF</a>
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

    /**
     * Possible initialization functions.
     */
    public enum Initialisation {
        /**
         * Initialize centroids by choosing uniformly at random from the data
         * points.
         */
        RANDOM,
        /**
         * KMeans++ initialisation.
         */
        PLUSPLUS
    }

    @Config(mandatory = true, description = "Number of centroids (i.e. the \"k\" in k-means).")
    private int centroids;

    @Config(mandatory = true, description = "The number of iterations to run.")
    private int iterations;

    @Config(mandatory = true, description = "The distance function to use.")
    private Distance distanceType;

    @Config(mandatory = true, description = "The centroid initialisation method to use.")
    private Initialisation initialisationType;

    @Config(description = "The number of threads to use for training.")
    private int numThreads = 1;

    @Config(mandatory = true, description = "The seed to use for the RNG.")
    private long seed;

    private SplittableRandom rng;

    private int trainInvocationCounter;

    /**
     * for olcut.
     */
    private KMeansTrainer() {
    }

    /**
     * Constructs a K-Means trainer using the supplied parameters.
     *
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param distanceType The distance function.
     * @param initialisationType The centroid initialization method.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public KMeansTrainer(int centroids, int iterations, Distance distanceType, Initialisation initialisationType, int numThreads, long seed) {
        this.centroids = centroids;
        this.iterations = iterations;
        this.distanceType = distanceType;
        this.initialisationType = initialisationType;
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
        synchronized (this) {
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();

        ForkJoinPool fjp = new ForkJoinPool(numThreads);

        int[] oldCentre = new int[examples.size()];
        SparseVector[] data = new SparseVector[examples.size()];
        double[] weights = new double[examples.size()];
        int n = 0;
        for (Example<ClusterID> example : examples) {
            weights[n] = example.getWeight();
            data[n] = SparseVector.createSparseVector(example, featureMap, false);
            oldCentre[n] = -1;
            n++;
        }

        DenseVector[] centroidVectors;
        switch (initialisationType) {
            case RANDOM:
                centroidVectors = initialiseRandomCentroids(centroids, featureMap, localRNG);
                break;
            case PLUSPLUS:
                centroidVectors = initialisePlusPlusCentroids(centroids, data, featureMap, localRNG, distanceType);
                break;
            default:
                throw new IllegalStateException("Unknown initialisation" + initialisationType);
        }

        Map<Integer, List<Integer>> clusterAssignments = new HashMap<>();
        for (int i = 0; i < centroids; i++) {
            clusterAssignments.put(i, Collections.synchronizedList(new ArrayList<>()));
        }

        boolean converged = false;

        for (int i = 0; (i < iterations) && !converged; i++) {
            //logger.log(Level.INFO,"Beginning iteration " + i);
            AtomicInteger changeCounter = new AtomicInteger(0);

            for (Entry<Integer, List<Integer>> e : clusterAssignments.entrySet()) {
                e.getValue().clear();
            }

            // E step
            Stream<SparseVector> vecStream = Arrays.stream(data);
            Stream<Integer> intStream = IntStream.range(0, data.length).boxed();
            Stream<IntAndVector> eStream;
            if (numThreads > 1) {
                eStream = StreamUtil.boundParallelism(StreamUtil.zip(intStream, vecStream, IntAndVector::new).parallel());
            } else {
                eStream = StreamUtil.zip(intStream, vecStream, IntAndVector::new);
            }
            try {
                fjp.submit(() -> eStream.forEach((IntAndVector e) -> {
                    double minDist = Double.POSITIVE_INFINITY;
                    int clusterID = -1;
                    int id = e.idx;
                    SparseVector vector = e.vector;
                    for (int j = 0; j < centroids; j++) {
                        DenseVector cluster = centroidVectors[j];
                        double distance = getDistance(cluster, vector, distanceType);
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
                throw new RuntimeException("Parallel execution failed", e);
            }
            //logger.log(Level.INFO, "E step completed. " + changeCounter.get() + " words updated.");

            mStep(fjp, centroidVectors, clusterAssignments, data, weights);

            logger.log(Level.INFO, "Iteration " + i + " completed. " + changeCounter.get() + " examples updated.");

            if (changeCounter.get() == 0) {
                converged = true;
                logger.log(Level.INFO, "K-Means converged at iteration " + i);
            }
        }

        Map<Integer, MutableLong> counts = new HashMap<>();
        for (Entry<Integer, List<Integer>> e : clusterAssignments.entrySet()) {
            counts.put(e.getKey(), new MutableLong(e.getValue().size()));
        }

        ImmutableOutputInfo<ClusterID> outputMap = new ImmutableClusteringInfo(counts);

        ModelProvenance provenance = new ModelProvenance(KMeansModel.class.getName(), OffsetDateTime.now(),
                examples.getProvenance(), trainerProvenance, runProvenance);

        return new KMeansModel("", provenance, featureMap, outputMap, centroidVectors, distanceType);
    }

    @Override
    public KMeansModel train(Dataset<ClusterID> dataset) {
        return train(dataset, Collections.emptyMap());
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    /**
     * Initialisation method called at the start of each train call when using the default centroid initialisation.
     * Centroids are initialised using a uniform random sample from the feature domain.
     *
     * @param centroids  The number of centroids to create.
     * @param featureMap The feature map to use for centroid sampling.
     * @param rng The RNG to use.
     * @return A {@link DenseVector} array of centroids.
     */
    private static DenseVector[] initialiseRandomCentroids(int centroids, ImmutableFeatureMap featureMap,
                                                           SplittableRandom rng) {
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

    /**
     * Initialisation method called at the start of each train call when using kmeans++ centroid initialisation.
     *
     * @param centroids The number of centroids to create.
     * @param data The dataset of {@link SparseVector} to use.
     * @param featureMap The feature map to use for centroid sampling.
     * @param rng The RNG to use.
     * @return A {@link DenseVector} array of centroids.
     */
    private static DenseVector[] initialisePlusPlusCentroids(int centroids, SparseVector[] data,
                                                             ImmutableFeatureMap featureMap, SplittableRandom rng,
                                                             Distance distanceType) {
        if (centroids > data.length) {
            throw new IllegalArgumentException("The number of centroids may not exceed the number of samples.");
        }

        int numFeatures = featureMap.size();
        double[] minDistancePerVector = new double[data.length];
        Arrays.fill(minDistancePerVector, Double.POSITIVE_INFINITY);

        double[] squaredMinDistance = new double[data.length];
        double[] probabilities = new double[data.length];
        DenseVector[] centroidVectors = new DenseVector[centroids];

        // set first centroid randomly from the data
        centroidVectors[0] = getRandomCentroidFromData(data, numFeatures, rng);

        // Set each uninitialised centroid remaining
        for (int i = 1; i < centroids; i++) {
            DenseVector prevCentroid = centroidVectors[i - 1];

            // go through every vector and see if the min distance to the
            // newest centroid is smaller than previous min distance for vec
            for (int j = 0; j < data.length; j++) {
                SparseVector curVec = data[j];
                double tempDistance = getDistance(prevCentroid, curVec, distanceType);
                minDistancePerVector[j] = Math.min(minDistancePerVector[j], tempDistance);
            }

            // square the distances and get total for normalization
            double total = 0.0;
            for (int j = 0; j < data.length; j++) {
                squaredMinDistance[j] = minDistancePerVector[j] * minDistancePerVector[j];
                total += squaredMinDistance[j];
            }

            // compute probabilities as p[i] = D(xi)^2 / sum(D(x)^2)
            for (int j = 0; j < probabilities.length; j++) {
                probabilities[j] = squaredMinDistance[j] / total;
            }

            // sample from probabilities to get the new centroid from data
            double[] cdf = Util.generateCDF(probabilities);
            int idx = Util.sampleFromCDF(cdf, rng);
            centroidVectors[i] = sparseToDense(data[idx], numFeatures);
        }
        return centroidVectors;
    }

    /**
     * Randomly select a piece of data as the starting centroid.
     *
     * @param data The dataset of {@link SparseVector} to use.
     * @param numFeatures The number of features.
     * @param rng The RNG to use.
     * @return A {@link DenseVector} representing a centroid.
     */
    private static DenseVector getRandomCentroidFromData(SparseVector[] data,
                                                         int numFeatures, SplittableRandom rng) {
        int rand_idx = rng.nextInt(data.length);
        return sparseToDense(data[rand_idx], numFeatures);
    }

    /**
     * Create a {@link DenseVector} from the data contained in a
     * {@link SparseVector}.
     *
     * @param vec The {@link SparseVector} to be transformed.
     * @param numFeatures The number of features.
     * @return A {@link DenseVector} containing the information from vec.
     */
    private static DenseVector sparseToDense(SparseVector vec, int numFeatures) {
        DenseVector dense = new DenseVector(numFeatures);
        dense.intersectAndAddInPlace(vec);
        return dense;
    }

    /**
     *
     * @param cluster A {@link DenseVector} representing a centroid.
     * @param vector A {@link SGDVector} representing an example.
     * @param distanceType The distance metric to employ.
     * @return A double representing the distance from vector to centroid.
     */
    private static double getDistance(DenseVector cluster, SGDVector vector,
                                      Distance distanceType) {
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
        return distance;
    }

    protected void mStep(ForkJoinPool fjp, DenseVector[] centroidVectors, Map<Integer, List<Integer>> clusterAssignments, SparseVector[] data, double[] weights) {
        // M step
        Stream<Entry<Integer, List<Integer>>> mStream;
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
                    newCentroid.intersectAndAddInPlace(data[idx], (double f) -> f * weights[idx]);
                    counter++;
                }
                if (counter > 0) {
                    newCentroid.scaleInPlace(1.0 / counter);
                }
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Parallel execution failed", e);
        }
    }

    @Override
    public String toString() {
        return "KMeansTrainer(centroids=" + centroids + ",distanceType=" + distanceType + ",seed=" + seed + ",numThreads=" + numThreads + ")";
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
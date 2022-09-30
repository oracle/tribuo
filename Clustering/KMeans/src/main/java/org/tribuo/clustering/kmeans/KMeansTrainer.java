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

package org.tribuo.clustering.kmeans;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
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
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.util.Util;

import java.security.AccessController;
import java.security.PrivilegedAction;
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
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
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
 * The train method will instantiate dense examples as dense vectors, speeding up the computation.
 * <p>
 * Note parallel training uses a {@link ForkJoinPool} which requires that the Tribuo codebase
 * is given the "modifyThread" and "modifyThreadGroup" privileges when running under a
 * {@link java.lang.SecurityManager}.
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

    // Thread factory for the FJP, to allow use with OpenSearch's SecureSM
    private static final CustomForkJoinWorkerThreadFactory THREAD_FACTORY = new CustomForkJoinWorkerThreadFactory();

    /**
     * Possible distance functions.
     * @deprecated
     * This Enum is deprecated in version 4.3, replaced by {@link DistanceType}
     */
    @Deprecated
    public enum Distance {
        /**
         * Euclidean (or l2) distance.
         */
        EUCLIDEAN(DistanceType.L2),
        /**
         * Cosine similarity as a distance measure.
         */
        COSINE(DistanceType.COSINE),
        /**
         * L1 (or Manhattan) distance.
         */
        L1(DistanceType.L1);

        private final DistanceType distanceType;

        Distance(DistanceType distanceType) {
            this.distanceType = distanceType;
        }

        /**
         * Returns the {@link DistanceType} mapping for the enumeration's value.
         *
         * @return distanceType The {@link DistanceType} value.
         */
        public DistanceType getDistanceType() {
            return distanceType;
        }
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

    @Config(mandatory = true, description = "Number of centroids (i.e., the \"k\" in k-means).")
    private int centroids;

    @Config(mandatory = true, description = "The number of iterations to run.")
    private int iterations;

    @Deprecated
    @Config(description = "The distance function to use. This is now deprecated.")
    private Distance distanceType;

    @Config(description = "The distance function to use.")
    private org.tribuo.math.distance.Distance dist;

    @Config(description = "The centroid initialisation method to use.")
    private Initialisation initialisationType = Initialisation.RANDOM;

    @Config(description = "The number of threads to use for training.")
    private int numThreads = 1;

    @Config(mandatory = true, description = "The seed to use for the RNG.")
    private long seed;

    private SplittableRandom rng;

    private int trainInvocationCounter;

    /**
     * for olcut.
     */
    private KMeansTrainer() { }

    /**
     * Constructs a K-Means trainer using the supplied parameters and the default random initialisation.
     * @deprecated
     * This Constructor is deprecated in version 4.3.
     *
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param distanceType The distance function.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    @Deprecated
    public KMeansTrainer(int centroids, int iterations, Distance distanceType, int numThreads, long seed) {
        this(centroids,iterations,distanceType,Initialisation.RANDOM,numThreads,seed);
    }

    /**
     * Constructs a K-Means trainer using the supplied parameters and the default random initialisation.
     *
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param dist The distance function.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public KMeansTrainer(int centroids, int iterations, org.tribuo.math.distance.Distance dist, int numThreads, long seed) {
        this(centroids,iterations,dist,Initialisation.RANDOM,numThreads,seed);
    }

    /**
     * Constructs a K-Means trainer using the supplied parameters.
     * @deprecated
     * This Constructor is deprecated in version 4.3.
     *
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param distanceType The distance function.
     * @param initialisationType The centroid initialization method.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    @Deprecated
    public KMeansTrainer(int centroids, int iterations, Distance distanceType, Initialisation initialisationType, int numThreads, long seed) {
        this(centroids, iterations, distanceType.getDistanceType().getDistance(), initialisationType, numThreads, seed);
    }

    /**
     * Constructs a K-Means trainer using the supplied parameters.
     *
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param dist The distance function.
     * @param initialisationType The centroid initialization method.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public KMeansTrainer(int centroids, int iterations, org.tribuo.math.distance.Distance dist, Initialisation initialisationType, int numThreads, long seed) {
        this.centroids = centroids;
        this.iterations = iterations;
        this.dist = dist;
        this.initialisationType = initialisationType;
        this.numThreads = numThreads;
        this.seed = seed;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        this.rng = new SplittableRandom(seed);

        if (this.distanceType != null) {
            if (this.dist != null) {
                throw new PropertyException("dist", "Both dist and distanceType must not both be set.");
            } else {
                this.dist = this.distanceType.getDistanceType().getDistance();
                this.distanceType = null;
            }
        }

        if (centroids < 1) {
            throw new PropertyException("centroids", "Centroids must be positive, found " + centroids);
        }
    }

    @Override
    public KMeansModel train(Dataset<ClusterID> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public KMeansModel train(Dataset<ClusterID> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        // Creates a new local RNG and adds one to the invocation count.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        synchronized (this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();

        int[] oldCentre = new int[examples.size()];
        SGDVector[] data = new SGDVector[examples.size()];
        double[] weights = new double[examples.size()];
        int n = 0;
        for (Example<ClusterID> example : examples) {
            weights[n] = example.getWeight();
            if (example.size() == featureMap.size()) {
                data[n] = DenseVector.createDenseVector(example, featureMap, false);
            } else {
                data[n] = SparseVector.createSparseVector(example, featureMap, false);
            }
            oldCentre[n] = -1;
            n++;
        }

        DenseVector[] centroidVectors;
        switch (initialisationType) {
            case RANDOM:
                centroidVectors = initialiseRandomCentroids(centroids, featureMap, localRNG);
                break;
            case PLUSPLUS:
                centroidVectors = initialisePlusPlusCentroids(centroids, data, localRNG, dist);
                break;
            default:
                throw new IllegalStateException("Unknown initialisation" + initialisationType);
        }

        Map<Integer, List<Integer>> clusterAssignments = new HashMap<>();
        boolean parallel = numThreads > 1;
        for (int i = 0; i < centroids; i++) {
            clusterAssignments.put(i, parallel ? Collections.synchronizedList(new ArrayList<>()) : new ArrayList<>());
        }

        AtomicInteger changeCounter = new AtomicInteger(0);
        Consumer<IntAndVector> eStepFunc = (IntAndVector e) -> {
            double minDist = Double.POSITIVE_INFINITY;
            int clusterID = -1;
            int id = e.idx;
            SGDVector vector = e.vector;
            for (int j = 0; j < centroids; j++) {
                DenseVector cluster = centroidVectors[j];
                double distance = dist.computeDistance(cluster, vector);
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
        };

        boolean converged = false;
        ForkJoinPool fjp = null;
        try {
            if (parallel) {
                if (System.getSecurityManager() == null) {
                    fjp = new ForkJoinPool(numThreads);
                } else {
                    fjp = new ForkJoinPool(numThreads, THREAD_FACTORY, null, false);
                }
            }
            for (int i = 0; (i < iterations) && !converged; i++) {
                logger.log(Level.FINE,"Beginning iteration " + i);
                changeCounter.set(0);

                for (Entry<Integer, List<Integer>> e : clusterAssignments.entrySet()) {
                    e.getValue().clear();
                }

                // E step
                Stream<SGDVector> vecStream = Arrays.stream(data);
                Stream<Integer> intStream = IntStream.range(0, data.length).boxed();
                Stream<IntAndVector> zipStream = StreamUtil.zip(intStream, vecStream, IntAndVector::new);
                if (parallel) {
                    Stream<IntAndVector> parallelZipStream = StreamUtil.boundParallelism(zipStream.parallel());
                    try {
                        fjp.submit(() -> parallelZipStream.forEach(eStepFunc)).get();
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException("Parallel execution failed", e);
                    }
                } else {
                    zipStream.forEach(eStepFunc);
                }
                logger.log(Level.FINE, "E step completed. " + changeCounter.get() + " words updated.");

                mStep(fjp, centroidVectors, clusterAssignments, data, weights);

                logger.log(Level.INFO, "Iteration " + i + " completed. " + changeCounter.get() + " examples updated.");

                if (changeCounter.get() == 0) {
                    converged = true;
                    logger.log(Level.INFO, "K-Means converged at iteration " + i);
                }
            }
        } finally {
            if (fjp != null) {
                fjp.shutdown();
            }
        }

        Map<Integer, MutableLong> counts = new HashMap<>();
        for (Entry<Integer, List<Integer>> e : clusterAssignments.entrySet()) {
            counts.put(e.getKey(), new MutableLong(e.getValue().size()));
        }

        ImmutableOutputInfo<ClusterID> outputMap = new ImmutableClusteringInfo(counts);

        ModelProvenance provenance = new ModelProvenance(KMeansModel.class.getName(), OffsetDateTime.now(),
                examples.getProvenance(), trainerProvenance, runProvenance);

        return new KMeansModel("k-means-model", provenance, featureMap, outputMap, centroidVectors, dist);
    }

    @Override
    public KMeansModel train(Dataset<ClusterID> dataset) {
        return train(dataset, Collections.emptyMap());
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        rng = new SplittableRandom(seed);

        for (trainInvocationCounter = 0; trainInvocationCounter < invocationCount; trainInvocationCounter++){
            SplittableRandom localRNG = rng.split();
        }

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
     * @param data The dataset of {@link SGDVector} to use.
     * @param rng The RNG to use.
     * @param dist The distance function.
     * @return A {@link DenseVector} array of centroids.
     */
    private static DenseVector[] initialisePlusPlusCentroids(int centroids, SGDVector[] data, SplittableRandom rng,
                                                             org.tribuo.math.distance.Distance dist) {
        if (centroids > data.length) {
            throw new IllegalArgumentException("The number of centroids may not exceed the number of samples.");
        }

        double[] minDistancePerVector = new double[data.length];
        Arrays.fill(minDistancePerVector, Double.POSITIVE_INFINITY);

        double[] squaredMinDistance = new double[data.length];
        double[] probabilities = new double[data.length];
        DenseVector[] centroidVectors = new DenseVector[centroids];

        // set first centroid randomly from the data
        centroidVectors[0] = getRandomCentroidFromData(data, rng);

        // Set each uninitialised centroid remaining
        for (int i = 1; i < centroids; i++) {
            DenseVector prevCentroid = centroidVectors[i - 1];

            // go through every vector and see if the min distance to the
            // newest centroid is smaller than previous min distance for vec
            for (int j = 0; j < data.length; j++) {
                double tempDistance = dist.computeDistance(prevCentroid, data[j]);
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
            centroidVectors[i] = DenseVector.createDenseVector(data[idx].toArray());
        }
        return centroidVectors;
    }

    /**
     * Randomly select a piece of data as the starting centroid.
     *
     * @param data The dataset of {@link SparseVector} to use.
     * @param rng The RNG to use.
     * @return A {@link DenseVector} representing a centroid.
     */
    private static DenseVector getRandomCentroidFromData(SGDVector[] data, SplittableRandom rng) {
        int randIdx = rng.nextInt(data.length);
        return DenseVector.createDenseVector(data[randIdx].toArray());
    }

    /**
     * Runs the mStep, writing to the {@code centroidVectors} array.
     * <p>
     * Note in 4.2 this method changed signature slightly, and overrides of the old
     * version will not match.
     * @param fjp The ForkJoinPool to run the computation in if it should be executed in parallel.
     *            If the fjp is null then the computation is executed sequentially on the main thread.
     * @param centroidVectors The centroid vectors to write out.
     * @param clusterAssignments The current cluster assignments.
     * @param data The data points.
     * @param weights The example weights.
     */
    protected void mStep(ForkJoinPool fjp, DenseVector[] centroidVectors, Map<Integer, List<Integer>> clusterAssignments, SGDVector[] data, double[] weights) {
        // M step
        Consumer<Entry<Integer, List<Integer>>> mStepFunc = (e) -> {
            DenseVector newCentroid = centroidVectors[e.getKey()];
            newCentroid.fill(0.0);

            double weightSum = 0.0;
            for (Integer idx : e.getValue()) {
                newCentroid.intersectAndAddInPlace(data[idx], (double f) -> f * weights[idx]);
                weightSum += weights[idx];
            }
            if (weightSum != 0.0) {
                newCentroid.scaleInPlace(1.0 / weightSum);
            }
        };

        Stream<Entry<Integer, List<Integer>>> mStream = clusterAssignments.entrySet().stream();
        if (fjp != null) {
            Stream<Entry<Integer, List<Integer>>> parallelMStream = StreamUtil.boundParallelism(mStream.parallel());
            try {
                fjp.submit(() -> parallelMStream.forEach(mStepFunc)).get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Parallel execution failed", e);
            }
        } else {
            mStream.forEach(mStepFunc);
        }
    }

    @Override
    public String toString() {
        return "KMeansTrainer(centroids=" + centroids + ",distance=" + dist + ",seed=" + seed + ",numThreads=" + numThreads + ", initialisationType=" + initialisationType + ")";
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
        final SGDVector vector;

        /**
         * Constructs an index and vector tuple.
         * @param idx The index.
         * @param vector The vector.
         */
        public IntAndVector(int idx, SGDVector vector) {
            this.idx = idx;
            this.vector = vector;
        }
    }

    /**
     * Used to allow FJPs to work with OpenSearch's SecureSM.
     */
    private static final class CustomForkJoinWorkerThreadFactory implements ForkJoinPool.ForkJoinWorkerThreadFactory {
        public final ForkJoinWorkerThread newThread(ForkJoinPool pool) {
            return AccessController.doPrivileged((PrivilegedAction<ForkJoinWorkerThread>) () -> new ForkJoinWorkerThread(pool) {});
        }
    }
}
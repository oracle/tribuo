/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.clustering.gmm;

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
import org.tribuo.WeightedExamples;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ImmutableClusteringInfo;
import org.tribuo.math.distance.L2Distance;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A Gaussian Mixture Model trainer, which generates a GMM clustering of the supplied
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
public class GMMTrainer implements Trainer<ClusterID>, WeightedExamples {
    private static final Logger logger = Logger.getLogger(GMMTrainer.class.getName());

    public enum CovarianceType {
        /**
         * Full covariance.
         */
        FULL,
        /**
         * Diagonal covariance.
         */
        DIAGONAL,
        /**
         * Spherical covariance.
         */
        SPHERICAL
    }

    /**
     * Possible initialization functions.
     */
    public enum Initialisation {
        /**
         * Initialize Gaussians by choosing uniformly at random from the data
         * points.
         */
        RANDOM,
        /**
         * KMeans++ initialisation.
         */
        PLUSPLUS
    }

    @Config(mandatory = true, description = "Number of centroids.")
    private int centroids;

    @Config(mandatory = true, description = "The number of iterations to run.")
    private int iterations;

    @Config(description = "The convergence threshold.")
    private double convergenceTolerance = 1e-3f;

    @Config(description = "The type of covariance matrix to fit.")
    private CovarianceType covarianceType = CovarianceType.DIAGONAL;

    @Config(description = "The centroid initialisation method to use.")
    private Initialisation initialisationType = Initialisation.RANDOM;

    @Config(description = "The number of threads to use for training.")
    private int numThreads = 1;

    @Config(mandatory = true, description = "The seed to use for the RNG.")
    private long seed;

    private SplittableRandom rng;

    private int trainInvocationCounter;

    private static final L2Distance plusPlusDistance = new L2Distance();

    /**
     * for olcut.
     */
    private GMMTrainer() { }

    /**
     * Constructs a K-Means trainer using the supplied parameters and the default random initialisation.
     *
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public GMMTrainer(int centroids, int iterations, int numThreads, long seed) {
        this(centroids,iterations,CovarianceType.DIAGONAL,Initialisation.RANDOM,1e-3,numThreads,seed);
    }

    /**
     * Constructs a K-Means trainer using the supplied parameters.
     *
     * @param centroids The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param initialisationType The centroid initialization method.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public GMMTrainer(int centroids, int iterations, CovarianceType covarianceType, Initialisation initialisationType, double tolerance, int numThreads, long seed) {
        this.centroids = centroids;
        this.iterations = iterations;
        this.covarianceType = covarianceType;
        this.initialisationType = initialisationType;
        this.convergenceTolerance = tolerance;
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

        if (centroids < 1) {
            throw new PropertyException("centroids", "Centroids must be positive, found " + centroids);
        }
    }

    @Override
    public GaussianMixtureModel train(Dataset<ClusterID> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public GaussianMixtureModel train(Dataset<ClusterID> examples, Map<String, Provenance> runProvenance, int invocationCount) {
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
            n++;
        }

        DenseMatrix responsibilities = new DenseMatrix(examples.size(), centroids);
        DenseVector[] meanVectors = switch (initialisationType) {
            case RANDOM -> initialiseRandomCentroids(centroids, featureMap, localRNG);
            case PLUSPLUS -> initialisePlusPlusCentroids(centroids, data, localRNG);
        };
        DenseMatrix[] covarianceMatrices = new DenseMatrix[centroids];
        DenseMatrix.CholeskyFactorization[] precisionFactorizations = new DenseMatrix.CholeskyFactorization[centroids];
        DenseVector mixingDistribution = new DenseVector(centroids);

        boolean parallel = numThreads > 1;

        Consumer<SGDVector> eStepFunc = (SGDVector e) -> {
            double minDist = Double.POSITIVE_INFINITY;
            for (int j = 0; j < centroids; j++) {
                DenseVector cluster = meanVectors[j];
                double distance = dist.computeDistance(cluster, e);
                if (distance < minDist) {
                    minDist = distance;
                }
            }
        };

        double oldLowerBound = Double.NEGATIVE_INFINITY;
        double newLowerBound;
        boolean converged = false;
        ForkJoinPool fjp = null;
        try {
            if (parallel) {
                fjp = new ForkJoinPool(numThreads);
            }
            for (int i = 0; (i < iterations) && !converged; i++) {
                logger.log(Level.FINE,"Beginning iteration " + i);

                // E step
                Stream<SGDVector> vecStream = Arrays.stream(data);
                if (parallel) {
                    try {
                        fjp.submit(() -> vecStream.parallel().forEach(eStepFunc)).get();
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException("Parallel execution failed", e);
                    }
                } else {
                    vecStream.forEach(eStepFunc);
                }
                logger.log(Level.FINE, i + "th e step completed.");

                // M step
                mStep(fjp, responsibilities, meanVectors, covarianceMatrices, mixingDistribution, precisionFactorizations, data, weights);
                logger.log(Level.FINE, i + "th m step completed.");

                // Compute log likelihood bound
                newLowerBound = computeLowerBound();

                logger.log(Level.INFO, "Iteration " + i + " completed.");

                if (newLowerBound - oldLowerBound < convergenceTolerance) {
                    converged = true;
                    logger.log(Level.INFO, "GMM converged at iteration " + i);
                }

                oldLowerBound = newLowerBound;
            }
        } finally {
            if (fjp != null) {
                fjp.shutdown();
            }
        }

        Map<Integer, MutableLong> counts = new HashMap<>();
        for (int i = 0; i < examples.size(); i++) {
            int idx = responsibilities.getRow(i).argmax();
            var count = counts.computeIfAbsent(idx, k -> new MutableLong());
            count.increment();
        }

        ImmutableOutputInfo<ClusterID> outputMap = new ImmutableClusteringInfo(counts);

        ModelProvenance provenance = new ModelProvenance(GaussianMixtureModel.class.getName(), OffsetDateTime.now(),
                examples.getProvenance(), trainerProvenance, runProvenance);

        return new GaussianMixtureModel("gaussian-mixture-model", provenance, featureMap, outputMap,
                meanVectors, covarianceMatrices, mixingDistribution);
    }

    @Override
    public GaussianMixtureModel train(Dataset<ClusterID> dataset) {
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
     * @return A {@link DenseVector} array of centroids.
     */
    private static DenseVector[] initialisePlusPlusCentroids(int centroids, SGDVector[] data, SplittableRandom rng) {
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
                double tempDistance = plusPlusDistance.computeDistance(prevCentroid, data[j]);
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
     * @param fjp The ForkJoinPool to run the computation in if it should be executed in parallel.
     *            If the fjp is null then the computation is executed sequentially.
     * @param centroidVectors The centroid vectors to write out.
     * @param data The data points.
     * @param weights The example weights.
     */
    protected void mStep(ForkJoinPool fjp, DenseVector[] centroidVectors, DenseMatrix[] covarianceMatrices,
                         DenseMatrix.CholeskyFactorization[] precisionFactorizations, DenseVector mixingDistribution, SGDVector[] data, double[] weights) {
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
            try {
                fjp.submit(() -> mStream.parallel().forEach(mStepFunc)).get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Parallel execution failed", e);
            }
        } else {
            mStream.forEach(mStepFunc);
        }
    }

    @Override
    public String toString() {
        return "GMMTrainer(centroids=" + centroids + ",seed=" + seed + ",numThreads=" + numThreads + ", initialisationType=" + initialisationType + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

}
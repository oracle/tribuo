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
import org.tribuo.math.distributions.MultivariateNormalDistribution;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A Gaussian Mixture Model trainer, which generates a GMM clustering of the supplied
 * data. The model finds the Gaussians, and then predict needs to be
 * called to infer the cluster assignments for the input data.
 * <p>
 * It's slightly contorted to fit the Tribuo Trainer and Model API, as the cluster assignments
 * can only be retrieved from the model after training, and require re-evaluating each example.
 * <p>
 * The Trainer has a selectable number of threads used in the training step.
 * The thread pool is local to an invocation of train, so there can be multiple concurrent trainings.
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
public class GMMTrainer implements Trainer<ClusterID> {
    private static final Logger logger = Logger.getLogger(GMMTrainer.class.getName());

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

    @Config(mandatory = true, description = "Number of Gaussians to fit.")
    private int numGaussians;

    @Config(mandatory = true, description = "The number of iterations to run.")
    private int iterations;

    @Config(description = "The convergence threshold.")
    private double convergenceTolerance = 1e-3f;

    @Config(description = "The type of covariance matrix to fit.")
    private MultivariateNormalDistribution.CovarianceType covarianceType = MultivariateNormalDistribution.CovarianceType.DIAGONAL;

    @Config(description = "The cluster initialisation method to use.")
    private Initialisation initialisationType = Initialisation.RANDOM;

    @Config(description = "The number of threads to use for training.")
    private int numThreads = 1;

    @Config(mandatory = true, description = "The seed to use for the RNG.")
    private long seed;

    @Config(description = "Jitter to add to the covariance diagonal.")
    private double covJitter = 1e-6;

    private SplittableRandom rng;

    private int trainInvocationCounter;

    private static final L2Distance plusPlusDistance = new L2Distance();

    /**
     * for olcut.
     */
    private GMMTrainer() { }

    /**
     * Constructs a Gaussian Mixture Model trainer using the supplied parameters and the default random initialisation.
     *
     * @param numGaussians The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public GMMTrainer(int numGaussians, int iterations, int numThreads, long seed) {
        this(numGaussians,iterations, MultivariateNormalDistribution.CovarianceType.DIAGONAL,Initialisation.RANDOM,1e-3,numThreads,seed);
    }

    /**
     * Constructs a Gaussian Mixture Model trainer using the supplied parameters.
     *
     * @param numGaussians The number of centroids to use.
     * @param iterations The maximum number of iterations.
     * @param initialisationType The centroid initialization method.
     * @param numThreads The number of threads.
     * @param seed The random seed.
     */
    public GMMTrainer(int numGaussians, int iterations, MultivariateNormalDistribution.CovarianceType covarianceType, Initialisation initialisationType, double tolerance, int numThreads, long seed) {
        this.numGaussians = numGaussians;
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

        if (numGaussians < 1) {
            throw new PropertyException("centroids", "Centroids must be positive, found " + numGaussians);
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
        final int numFeatures = featureMap.size();

        DenseVector[] responsibilities = new DenseVector[examples.size()];
        SGDVector[] data = new SGDVector[examples.size()];
        int n = 0;
        for (Example<ClusterID> example : examples) {
            if (example.size() == numFeatures) {
                data[n] = DenseVector.createDenseVector(example, featureMap, false);
            } else {
                data[n] = SparseVector.createSparseVector(example, featureMap, false);
            }
            responsibilities[n] = new DenseVector(numGaussians);
            n++;
        }

        final DenseVector[] meanVectors = switch (initialisationType) {
            case RANDOM -> initialiseRandomCentroids(numGaussians, featureMap, localRNG);
            case PLUSPLUS -> initialisePlusPlusCentroids(numGaussians, data, localRNG);
        };
        Tensor[] covariances = new Tensor[numGaussians];
        final Tensor[] precision = new Tensor[numGaussians];
        final double[] determinant = new double[numGaussians];
        final DenseVector mixingDistribution = new DenseVector(numGaussians, 1.0/numGaussians);

        final Tensor covarianceJitter;
        if (covarianceType == MultivariateNormalDistribution.CovarianceType.FULL) {
            covarianceJitter = DenseMatrix.createIdentity(numFeatures);
            covarianceJitter.scaleInPlace(covJitter);
        } else {
            covarianceJitter = new DenseVector(numFeatures, covJitter);
        }

        for (int i = 0; i < numGaussians; i++) {
            determinant[i] = 1.0;
            switch (covarianceType) {
                case FULL -> {
                    covariances[i] = DenseMatrix.createIdentity(numFeatures);
                    precision[i] = DenseMatrix.createIdentity(numFeatures);
                }
                case DIAGONAL, SPHERICAL -> {
                    covariances[i] = new DenseVector(numFeatures, 1.0);
                    precision[i] = new DenseVector(numFeatures, 1.0);
                }
            }
        }

        boolean parallel = numThreads > 1;

        ToDoubleFunction<Vectors> eStepFunc = (Vectors e) -> {
            DenseVector curResponsibilities = e.responsibility;
            // compute log probs
            for (int i = 0; i < meanVectors.length; i++) {
                curResponsibilities.set(i, MultivariateNormalDistribution.logProbability(e.data, meanVectors[i], precision[i], determinant[i], covarianceType));
            }

            // add mixing distribution
            curResponsibilities.intersectAndAddInPlace(mixingDistribution, Math::log);

            // normalize log probabilities
            double sum = curResponsibilities.logSumExp();
            curResponsibilities.scalarAddInPlace(-sum);

            // exponentiate them
            curResponsibilities.foreachInPlace(Math::exp);

            return sum;
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
                double normSum;
                Stream<SGDVector> dataEStream = Arrays.stream(data);
                Stream<DenseVector> resEStream = Arrays.stream(responsibilities);
                Stream<Vectors> zipEStream = StreamUtil.zip(dataEStream, resEStream, Vectors::new);
                if (parallel) {
                    try {
                        normSum = fjp.submit(() -> zipEStream.parallel().mapToDouble(eStepFunc).sum()).get();
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException("Parallel execution failed", e);
                    }
                } else {
                    normSum = zipEStream.mapToDouble(eStepFunc).sum();
                }
                logger.log(Level.FINE, i + "th e step completed.");

                // compute lower bound
                newLowerBound = normSum / examples.size();

                // M step
                // compute new mixing distribution
                DenseVector zeroVector = new DenseVector(numGaussians);
                Stream<DenseVector> resStream = Arrays.stream(responsibilities);
                DenseVector newMixingDistribution;
                if (parallel) {
                    try {
                        newMixingDistribution = fjp.submit(() -> resStream.parallel().reduce(zeroVector, DenseVector::add)).get();
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException("Parallel execution failed", e);
                    }
                } else {
                    newMixingDistribution = resStream.parallel().reduce(zeroVector, DenseVector::add);
                }
                // add minimum value to ensure all values are positive
                newMixingDistribution.scalarAddInPlace(2e-15);

                // compute new means based on mixing distribution & positions
                for (int j = 0; j < numGaussians; j++) {
                    meanVectors[j].set(0);
                }
                // Manual matrix multiply here as things are stored as arrays of vectors
                // responsibilities[examples, gaussians], data[examples, features], means[gaussians, features]
                for (int j = 0; j < examples.size(); j++) {
                    DenseVector curResp = responsibilities[j];
                    SGDVector curExample = data[j];
                    for (VectorTuple v : curExample) {
                        for (int k = 0; k < numGaussians; k++) {
                            DenseVector curMean = meanVectors[k];
                            curMean.set(v.index, curMean.get(v.index) + v.value * curResp.get(k));
                        }
                    }
                }
                for (int j = 0; j < numGaussians; j++) {
                    meanVectors[j].scaleInPlace(1.0/newMixingDistribution.get(j));
                }

                // compute new covariances
                Stream<SGDVector> dataMStream = Arrays.stream(data);
                Stream<DenseVector> resMStream = Arrays.stream(responsibilities);
                Stream<Vectors> zipMStream = StreamUtil.zip(dataMStream, resMStream, Vectors::new);
                Supplier<Tensor[]> zeroTensor = switch (covarianceType) {
                    case FULL -> () -> {
                        Tensor[] output = new Tensor[numGaussians];
                        for (int j = 0; j < numGaussians; j++) {
                            output[j] = new DenseMatrix(numFeatures, numFeatures);
                        }
                        return output;
                    };
                    case DIAGONAL, SPHERICAL -> () -> {
                        Tensor[] output = new Tensor[numGaussians];
                        for (int j = 0; j < numGaussians; j++) {
                            output[j] = new DenseVector(numFeatures);
                        }
                        return output;
                    };
                };
                // Fix parallel behaviour
                BiConsumer<Tensor[], Vectors> mStep = switch (covarianceType) {
                    case FULL -> (Tensor[] input, Vectors v) -> {
                        for (int j = 0; j < numGaussians; j++) {
                            // Compute covariance contribution from current input
                            DenseMatrix curCov = (DenseMatrix) input[j];

                            DenseVector diff = (DenseVector) v.data.subtract(meanVectors[j]);
                            diff.scaleInPlace(v.responsibility.get(j) / newMixingDistribution.get(j));
                            curCov.intersectAndAddInPlace(diff.outer(diff));
                        }
                    };
                    case DIAGONAL -> (Tensor[] input, Vectors v) -> {
                        for (int j = 0; j < numGaussians; j++) {
                            // Compute covariance contribution from current input
                            DenseVector curCov = (DenseVector) input[j];
                            DenseVector diff = (DenseVector) v.data.subtract(meanVectors[j]);
                            diff.foreachInPlace(a -> a * a);
                            diff.scaleInPlace(v.responsibility.get(j) / newMixingDistribution.get(j));
                            curCov.intersectAndAddInPlace(diff);
                        }
                    };
                    case SPHERICAL -> (Tensor[] input, Vectors v) -> {
                        for (int j = 0; j < numGaussians; j++) {
                            // Compute covariance contribution from current input
                            DenseVector curCov = (DenseVector) input[j];
                            DenseVector diff = (DenseVector) v.data.subtract(meanVectors[j]);
                            diff.foreachInPlace(a -> a * a);
                            diff.scaleInPlace(v.responsibility.get(j) / newMixingDistribution.get(j));
                            double mean = diff.sum() / numFeatures;
                            diff.set(mean);
                            curCov.intersectAndAddInPlace(diff);
                        }
                    };
                };
                BiConsumer<Tensor[], Tensor[]> combineTensor = (Tensor[] a, Tensor[] b) -> {
                    for (int j = 0; j < a.length; j++) {
                        if (a[j] instanceof DenseMatrix aMat && b[j] instanceof DenseMatrix bMat) {
                            aMat.intersectAndAddInPlace(bMat);
                        } else if (a[j] instanceof DenseVector aVec && b[j] instanceof DenseVector bVec) {
                            aVec.intersectAndAddInPlace(bVec);
                        } else {
                            throw new IllegalStateException("Invalid types in reduce, expected both DenseMatrix or DenseVector, found " + a[j].getClass() + " and " + b[j].getClass());
                        }
                    }
                };
                if (parallel) {
                    try {
                        covariances = fjp.submit(() -> zipMStream.parallel().collect(zeroTensor, mStep, combineTensor)).get();
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException("Parallel execution failed", e);
                    }
                } else {
                    covariances = zipMStream.collect(zeroTensor, mStep, combineTensor);
                }

                // renormalize mixing distribution
                double mixingSum = newMixingDistribution.sum();
                newMixingDistribution.scaleInPlace(1/mixingSum);
                mixingDistribution.setElements(newMixingDistribution);

                // compute precisions
                switch (covarianceType) {
                    case FULL -> {
                        for (int j = 0; j < covariances.length; j++) {
                            DenseMatrix covMat = (DenseMatrix) covariances[j];
                            covMat.intersectAndAddInPlace(covarianceJitter);
                            Optional<DenseMatrix.CholeskyFactorization> optFact = covMat.choleskyFactorization();
                            if (optFact.isPresent()) {
                                DenseMatrix.CholeskyFactorization fact = optFact.get();
                                precision[j] = fact.inverse();
                                determinant[j] = fact.determinant();
                            } else {
                                throw new IllegalStateException("Failed to invert covariance matrix, cholesky didn't complete.");
                            }
                        }
                    }
                    case DIAGONAL, SPHERICAL -> {
                        for (int j = 0; j < covariances.length; j++) {
                            DenseVector covVec = (DenseVector) covariances[j];
                            covVec.intersectAndAddInPlace(covarianceJitter);
                            DenseVector preVec = (DenseVector) precision[j];
                            double tmp = 1;
                            for (int k = 0; k < preVec.size(); k++) {
                                double curVal = 1/Math.sqrt(covVec.get(k));
                                // Sets the value in the precision array.
                                preVec.set(k, curVal);
                                tmp *= covVec.get(k);
                            }
                            precision[j] = preVec;
                            determinant[j] = tmp;
                        }
                    }
                }

                logger.log(Level.FINE, i + "th m step completed.");

                logger.log(Level.INFO, "Iteration " + i + " completed.");

                if (Math.abs(newLowerBound - oldLowerBound) < convergenceTolerance) {
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
            int idx = responsibilities[i].argmax();
            var count = counts.computeIfAbsent(idx, k -> new MutableLong());
            count.increment();
        }

        ImmutableOutputInfo<ClusterID> outputMap = new ImmutableClusteringInfo(counts);

        ModelProvenance provenance = new ModelProvenance(GaussianMixtureModel.class.getName(), OffsetDateTime.now(),
                examples.getProvenance(), trainerProvenance, runProvenance);

        return new GaussianMixtureModel("gaussian-mixture-model", provenance, featureMap, outputMap,
                meanVectors, covariances, mixingDistribution, covarianceType);
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
            DenseVector newCentroid = new DenseVector(numFeatures);

            for (int j = 0; j < numFeatures; j++) {
                newCentroid.set(j, featureMap.get(j).uniformSample(rng));
            }

            centroidVectors[i] = newCentroid;
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

    @Override
    public String toString() {
        return "GMMTrainer(numGaussians=" + numGaussians + ",seed=" + seed + ",numThreads=" + numThreads + ", initialisationType=" + initialisationType + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    /**
     * Tuple of data and responsibility vectors.
     */
    record Vectors(SGDVector data, DenseVector responsibility) { }
}
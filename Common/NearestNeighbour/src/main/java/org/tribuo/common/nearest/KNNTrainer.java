/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.nearest;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.common.nearest.KNNModel.Backend;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.neighbour.NeighboursQueryFactory;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.Map;

/**
 * A {@link Trainer} for k-nearest neighbour models.
 */
public class KNNTrainer<T extends Output<T>> implements Trainer<T> {

    /**
     * The available distance functions.
     * @deprecated
     * This Enum is deprecated in version 4.3, replaced by {@link DistanceType}
     */
    @Deprecated
    public enum Distance {
        /**
         * L1 (or Manhattan) distance.
         */
        L1(DistanceType.L1),
        /**
         * L2 (or Euclidean) distance.
         */
        L2(DistanceType.L2),
        /**
         * Cosine similarity used as a distance measure.
         */
        COSINE(DistanceType.COSINE);

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

    @Deprecated
    @Config(description="The distance function used to measure nearest neighbours. This is now deprecated.")
    private Distance distance;

    @Config(description = "The distance function used to measure nearest neighbours.")
    private org.tribuo.math.distance.Distance dist;

    @Config(mandatory = true, description="The number of nearest neighbours to check.")
    private int k;

    @Config(mandatory = true, description="The combination function to aggregate the nearest neighbours.")
    private EnsembleCombiner<T> combiner;

    @Config(description="The number of threads to use for inference.")
    private int numThreads = 1;

    @Config(description="The threading model to use.")
    private Backend backend = Backend.THREADPOOL;

    @Config(description = "The nearest neighbour implementation factory to use.")
    private NeighboursQueryFactory neighboursQueryFactory;

    private int trainInvocationCount = 0;

    /**
     * For olcut.
     */
    private KNNTrainer() {}

    /**
     * Creates a K-NN trainer using the supplied parameters.
     * @param k The number of nearest neighbours to consider.
     * @param dist The distance function.
     * @param numThreads The number of threads to use.
     * @param combiner The combination function to aggregate the k predictions.
     * @param backend The computational backend.
     * @param nqFactoryType The nearest neighbour implementation factory to use.
     */
    public KNNTrainer(int k, org.tribuo.math.distance.Distance dist, int numThreads, EnsembleCombiner<T> combiner,
                      Backend backend, NeighboursQueryFactoryType nqFactoryType) {
        this.k = k;
        this.dist = dist;
        this.numThreads = numThreads;
        this.combiner = combiner;
        this.backend = backend;
        this.neighboursQueryFactory = NeighboursQueryFactoryType.getNeighboursQueryFactory(nqFactoryType, dist, numThreads);
        postConfig();
    }

    /**
     * Creates a K-NN trainer using the supplied parameters. {@link #neighboursQueryFactory} defaults to
     * {@link NeighboursBruteForceFactory}.
     * @deprecated
     * This Constructor is deprecated in version 4.3.
     *
     * @param k The number of nearest neighbours to consider.
     * @param distance The distance function.
     * @param numThreads The number of threads to use.
     * @param combiner The combination function to aggregate the k predictions.
     * @param backend The computational backend.
     */
    @Deprecated
    public KNNTrainer(int k, Distance distance, int numThreads, EnsembleCombiner<T> combiner, Backend backend) {
        this(k, distance.getDistanceType().getDistance(), numThreads, combiner, backend, NeighboursQueryFactoryType.BRUTE_FORCE);
    }

    /**
     * Creates a K-NN trainer using the supplied parameters.
     *
     * @param k The number of nearest neighbours to consider.
     * @param numThreads The number of threads to use.
     * @param combiner The combination function to aggregate the k predictions.
     * @param backend The computational backend.
     * @param neighboursQueryFactory The nearest neighbour implementation factory to use.
     */
    public KNNTrainer(int k, int numThreads, EnsembleCombiner<T> combiner,
                      Backend backend, NeighboursQueryFactory neighboursQueryFactory) {
        this.k = k;
        this.dist = neighboursQueryFactory.getDistance();
        this.numThreads = numThreads;
        this.combiner = combiner;
        this.backend = backend;
        this.neighboursQueryFactory = neighboursQueryFactory;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        if (k < 1) {
            throw new PropertyException("","k","k must be greater than 0");
        }

        if (this.distance != null) {
            if (this.dist != null) {
                throw new PropertyException("dist", "Both dist and distanceType must not both be set.");
            } else {
                this.dist = this.distance.getDistanceType().getDistance();
                this.distance = null;
            }
        }

        if (neighboursQueryFactory == null) {
            int numberThreads = (this.numThreads <= 0) ? 1 : this.numThreads;
            this.neighboursQueryFactory = new NeighboursBruteForceFactory(dist, numberThreads);
        } else {
            if (!this.dist.equals(neighboursQueryFactory.getDistance())) {
                throw new PropertyException("neighboursQueryFactory", "distType and its field on the " +
                    "NeighboursQueryFactory must be equal.");
            }
        }
    }

    @Override
    public Model<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
        return(train(examples, runProvenance, INCREMENT_INVOCATION_COUNT));
    }

    @Override
    public Model<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> labelIDMap = examples.getOutputIDInfo();

        @SuppressWarnings("unchecked") // generic array creation
        Pair<SGDVector,T>[] vectors = new Pair[examples.size()];

        int i = 0;
        for (Example<T> e : examples) {
            if (e.size() == featureIDMap.size()) {
                vectors[i] = new Pair<>(DenseVector.createDenseVector(e, featureIDMap, false),e.getOutput());
            } else {
                vectors[i] = new Pair<>(SparseVector.createSparseVector(e,featureIDMap,false),e.getOutput());
            }
            i++;
        }

        if(invocationCount != INCREMENT_INVOCATION_COUNT){
            setInvocationCount(invocationCount);
        }
        trainInvocationCount++;

        ModelProvenance provenance = new ModelProvenance(KNNModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), runProvenance);

        return new KNNModel<>(k+"nn",provenance, featureIDMap, labelIDMap, false, k, dist,
            numThreads, combiner, vectors, backend, neighboursQueryFactory);
    }

    @Override
    public String toString() {
        return "KNNTrainer(k="+k+",distance="+dist+",combiner="+combiner.toString()+",numThreads="+numThreads+")";
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCount;
    }

    @Override
    public void setInvocationCount(int invocationCount) {
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        this.trainInvocationCount = invocationCount;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}

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

import com.oracle.labs.mlrg.olcut.util.Pair;
import com.oracle.labs.mlrg.olcut.util.StreamUtil;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.common.nearest.KNNTrainer.Distance;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.neighbour.NeighboursQuery;
import org.tribuo.math.neighbour.NeighboursQueryFactory;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.Future;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A k-nearest neighbours model.
 * <p>
 * Note multi-threaded prediction uses a {@link ForkJoinPool} which requires that the Tribuo codebase
 * is given the "modifyThread" and "modifyThreadGroup" privileges when running under a
 * {@link java.lang.SecurityManager}.
 */
public class KNNModel<T extends Output<T>> extends Model<T> {

    private static final Logger logger = Logger.getLogger(KNNModel.class.getName());

    private static final long serialVersionUID = 1L;

    // Thread factory for the FJP, to allow use with OpenSearch's SecureSM
    private static final CustomForkJoinWorkerThreadFactory THREAD_FACTORY = new CustomForkJoinWorkerThreadFactory();

    /**
     * The parallel backend for batch predictions.
     */
    public enum Backend {
        /**
         * Uses the streams API for parallelism when scoring a batch of predictions.
         */
        STREAMS,
        /**
         * Uses a thread pool at the outer level (i.e., one thread per prediction).
         */
        THREADPOOL,
        /**
         * Uses a thread pool at the inner level (i.e., the whole thread pool works on each prediction).
         */
        INNERTHREADPOOL
    }

    private final Pair<SGDVector,T>[] vectors;

    private final int k;
    @Deprecated
    private Distance distance;

    // This is not final to support deserialization of older models. It will be final in a future version which doesn't
    // maintain serialization compatibility with 4.X.
    private DistanceType distType;

    private final int numThreads;

    private final Backend parallelBackend;

    private final EnsembleCombiner<T> combiner;

    // This is not final to support deserialization of older models. It will be final in a future version which doesn't
    // maintain serialization compatibility with 4.X.
    private NeighboursQueryFactory neighboursQueryFactory;

    private transient NeighboursQuery neighboursQuery;

    KNNModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
             boolean generatesProbabilities, int k, DistanceType distType, int numThreads, EnsembleCombiner<T> combiner,
             Pair<SGDVector,T>[] vectors, Backend backend, NeighboursQueryFactory neighboursQueryFactory) {
        super(name,provenance,featureIDMap,outputIDInfo,generatesProbabilities);
        this.k = k;
        this.distType = distType;
        this.numThreads = numThreads;
        this.combiner = combiner;
        this.parallelBackend = backend;
        this.vectors = vectors;
        this.neighboursQueryFactory = neighboursQueryFactory;
        this.neighboursQuery = neighboursQueryFactory.createNeighboursQuery(getSGDVectorArr());
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        SGDVector input;
        if (example.size() == featureIDMap.size()) {
            input = DenseVector.createDenseVector(example, featureIDMap, false);
        } else {
            input = SparseVector.createSparseVector(example, featureIDMap, false);
        }

        if (input.numActiveElements() == 0) {
            throw new IllegalArgumentException("No features found in Example " + example);
        }

        Function<Pair<SGDVector,T>, OutputDoublePair<T>> distanceFunc =
            (a) -> new OutputDoublePair<>(a.getB(),DistanceType.getDistance(a.getA(), input, distType));

        List<Prediction<T>> predictions;
        Stream<Pair<SGDVector,T>> stream = Stream.of(vectors);
        if (numThreads > 1) {
            ForkJoinPool fjp = System.getSecurityManager() == null ? new ForkJoinPool(numThreads) : new ForkJoinPool(numThreads, THREAD_FACTORY, null, false);
            try {
                predictions = fjp.submit(()->StreamUtil.boundParallelism(stream.parallel()).map(distanceFunc).sorted().limit(k).map((a) -> new Prediction<>(a.output, input.numActiveElements(), example)).collect(Collectors.toList())).get();
            } catch (InterruptedException | ExecutionException e) {
                logger.log(Level.SEVERE,"Exception when predicting in KNNModel",e);
                throw new IllegalStateException("Failed to process example in parallel",e);
            }
        } else {
            predictions = stream.map(distanceFunc).sorted().limit(k).map((a) -> new Prediction<>(a.output, input.numActiveElements(), example)).collect(Collectors.toList());
        }

        return combiner.combine(outputIDInfo,predictions);
    }

    /**
     * Uses the model to predict the output for multiple examples.
     * @param examples the examples to predict.
     * @return the results of the prediction, in the same order as the
     * examples.
     */
    @Override
    protected List<Prediction<T>> innerPredict(Iterable<Example<T>> examples) {
        if (numThreads > 1) {
            return innerPredictMultithreaded(examples);
        } else {
            List<Prediction<T>> predictions = new ArrayList<>();
            List<Prediction<T>> innerPredictions = new ArrayList<>();

            for (Example<T> example : examples) {
                innerPredictions.clear();
                SGDVector input;
                if (example.size() == featureIDMap.size()) {
                    input = DenseVector.createDenseVector(example, featureIDMap, false);
                } else {
                    input = SparseVector.createSparseVector(example, featureIDMap, false);
                }

                List<Pair<Integer, Double>> indexDistancePairList = neighboursQuery.query(input, k);

                for (Pair<Integer, Double> simplePair : indexDistancePairList) {
                    Pair<SGDVector,T> pair = vectors[simplePair.getA()];
                    innerPredictions.add(new Prediction<>(pair.getB(), input.numActiveElements(), example));
                }

                predictions.add(combiner.combine(outputIDInfo, innerPredictions));
            }
            return predictions;
        }
    }

    /**
     * Switches between the different multithreaded backends.
     * @param examples The examples to predict.
     * @return The predictions.
     */
    private List<Prediction<T>> innerPredictMultithreaded(Iterable<Example<T>> examples) {
        switch (parallelBackend) {
            case STREAMS:
                logger.log(Level.FINE, "Parallel backend - streams");
                return innerPredictStreams(examples);
            case THREADPOOL:
                logger.log(Level.FINE, "Parallel backend - threadpool");
                return innerPredictThreadPool(examples);
            case INNERTHREADPOOL:
                logger.log(Level.FINE, "Parallel backend - within example threadpool");
                return innerPredictWithinExampleThreadPool(examples);
            default:
                throw new IllegalArgumentException("Unknown backend " + parallelBackend);
        }
    }

    /**
     * Predicts using a FJP and the Streams API.
     * @param examples The examples to predict.
     * @return The predictions.
     */
    private List<Prediction<T>> innerPredictStreams(Iterable<Example<T>> examples) {
        List<Prediction<T>> predictions = new ArrayList<>();
        List<Prediction<T>> innerPredictions = null;
        ForkJoinPool fjp = System.getSecurityManager() == null ? new ForkJoinPool(numThreads) : new ForkJoinPool(numThreads, THREAD_FACTORY, null, false);
        for (Example<T> example : examples) {
            SGDVector input;
            if (example.size() == featureIDMap.size()) {
                input = DenseVector.createDenseVector(example, featureIDMap, false);
            } else {
                input = SparseVector.createSparseVector(example, featureIDMap, false);
            }

            Function<Pair<SGDVector, T>, OutputDoublePair<T>> distanceFunc =
                (a) -> new OutputDoublePair<>(a.getB(), DistanceType.getDistance(a.getA(), input, distType));

            Stream<Pair<SGDVector, T>> stream = Stream.of(vectors);
            try {
                innerPredictions = fjp.submit(() -> StreamUtil.boundParallelism(stream.parallel()).map(distanceFunc).sorted().limit(k).map((a) -> new Prediction<>(a.output, input.numActiveElements(), example)).collect(Collectors.toList())).get();
            } catch (InterruptedException | ExecutionException e) {
                logger.log(Level.SEVERE, "Exception when predicting in KNNModel", e);
            }

            predictions.add(combiner.combine(outputIDInfo, innerPredictions));
        }

        return predictions;
    }

    /**
     * Uses a thread pool, one thread per prediction.
     * @param examples The examples to predict.
     * @return The predictions.
     */
    private List<Prediction<T>> innerPredictThreadPool(Iterable<Example<T>> examples) {
        List<Prediction<T>> predictions = new ArrayList<>();

        ExecutorService pool = Executors.newFixedThreadPool(numThreads);

        List<Future<Prediction<T>>> futures = new ArrayList<>();

        for (Example<T> example : examples) {
            futures.add(pool.submit(() -> innerPredictOne(neighboursQuery,vectors,combiner,featureIDMap,outputIDInfo,k,example)));
        }

        try {
            for (Future<Prediction<T>> f : futures) {
                predictions.add(f.get());
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new IllegalStateException("Thread pool went bang",e);
        }

        pool.shutdown();

        return predictions;
    }

    /**
     * Uses a thread pool where the pool collaborates on each example (best for large training dataset sizes).
     * @param examples The examples to predict.
     * @return The predictions.
     */
    private List<Prediction<T>> innerPredictWithinExampleThreadPool(Iterable<Example<T>> examples) {

        List<Prediction<T>> predictions = new ArrayList<>();

        ExecutorService pool = Executors.newFixedThreadPool(numThreads);

        ThreadLocal<PriorityQueue<OutputDoublePair<T>>> queuePool = ThreadLocal.withInitial(() -> new PriorityQueue<>(k, (a,b) -> Double.compare(b.value, a.value)));

        for (Example<T> example : examples) {
            predictions.add(innerPredictThreadPool(pool,queuePool,distType,example));
        }

        pool.shutdown();

        return predictions;
    }

    private Prediction<T> innerPredictThreadPool(ExecutorService pool,
                                                 ThreadLocal<PriorityQueue<OutputDoublePair<T>>> queuePool,
                                                 DistanceType distType,
                                                 Example<T> example) {
        SparseVector vector = SparseVector.createSparseVector(example, featureIDMap, false);
        List<Future<List<OutputDoublePair<T>>>> futures = new ArrayList<>();

        for (int i = 0; i < numThreads; i++) {
            int start = i * (vectors.length / numThreads);
            int end = (i + 1) * (vectors.length / numThreads);
            futures.add(pool.submit(() -> innerPredictChunk(queuePool,vectors,start,end,distType,k,vector)));
        }

        PriorityQueue<OutputDoublePair<T>> queue = new PriorityQueue<>(k, (a,b) -> Double.compare(b.value, a.value));
        try {
            for (Future<List<OutputDoublePair<T>>> f : futures) {
                List<OutputDoublePair<T>> chunkOutputs = f.get();
                for (OutputDoublePair<T> curOutputPair : chunkOutputs) {
                    if (queue.size() < k) {
                        queue.offer(curOutputPair);
                    } else if (Double.compare(curOutputPair.value, queue.peek().value) < 0) {
                        queue.poll();
                        queue.offer(curOutputPair);
                    }
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new IllegalStateException("Thread pool went bang",e);
        }

        List<Prediction<T>> predictions = new ArrayList<>();

        for (OutputDoublePair<T> pair : queue) {
            predictions.add(new Prediction<>(pair.output,vector.numActiveElements(),example));
        }

        return combiner.combine(outputIDInfo,predictions);
    }

    private static <T extends Output<T>> List<OutputDoublePair<T>> innerPredictChunk(ThreadLocal<PriorityQueue<OutputDoublePair<T>>> queuePool,
                                                                            Pair<SGDVector,T>[] vectors,
                                                                            int start,
                                                                            int end,
                                                                            DistanceType distType,
                                                                            int k,
                                                                            SGDVector input) {
        PriorityQueue<OutputDoublePair<T>> queue = queuePool.get();
        queue.clear();

        end = Math.min(end, vectors.length);

        for (int i = start; i < end; i++) {
            double curDistance = DistanceType.getDistance(vectors[i].getA(), input, distType);

            if (queue.size() < k) {
                OutputDoublePair<T> newPair = new OutputDoublePair<>(vectors[i].getB(),curDistance);
                queue.offer(newPair);
            } else if (Double.compare(curDistance, queue.peek().value) < 0) {
                OutputDoublePair<T> pair = queue.poll();
                pair.output = vectors[i].getB();
                pair.value = curDistance;
                queue.offer(pair);
            }
        }

        return new ArrayList<>(queue);
    }

    private static <T extends Output<T>> Prediction<T> innerPredictOne(NeighboursQuery nq,
                                                                    Pair<SGDVector,T>[] vectors,
                                                                    EnsembleCombiner<T> combiner,
                                                                    ImmutableFeatureMap featureIDMap,
                                                                    ImmutableOutputInfo<T> outputIDInfo,
                                                                    int k,
                                                                    Example<T> example) {
        SGDVector vector;
        if (example.size() == featureIDMap.size()) {
            vector = DenseVector.createDenseVector(example, featureIDMap, false);
        } else {
            vector = SparseVector.createSparseVector(example, featureIDMap, false);
        }

        List<Pair<Integer, Double>> indexDistancePairList = nq.query(vector, k);

        List<Prediction<T>> localPredictions = new ArrayList<>();

        for (Pair<Integer, Double> simplePair : indexDistancePairList) {
            Pair<SGDVector,T> pair = vectors[simplePair.getA()];
            localPredictions.add(new Prediction<>(pair.getB(), vector.numActiveElements(), example));
        }

        return combiner.combine(outputIDInfo,localPredictions);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        return Optional.empty();
    }

    @SuppressWarnings("unchecked") // Generic array creation.
    @Override
    protected KNNModel<T> copy(String newName, ModelProvenance newProvenance) {
        Pair<SGDVector,T>[] vectorCopy = new Pair[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            vectorCopy[i] = new Pair<>(vectors[i].getA().copy(),vectors[i].getB().copy());
        }
        return new KNNModel<>(newName,newProvenance,featureIDMap,outputIDInfo,generatesProbabilities,k,distType,
            numThreads,combiner,vectorCopy,parallelBackend,neighboursQueryFactory);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        if (distType == null) {
            distType = distance.getDistanceType();
        }
        if (neighboursQueryFactory == null) {
            neighboursQueryFactory = new NeighboursBruteForceFactory(distType, numThreads);
        }
        neighboursQuery = neighboursQueryFactory.createNeighboursQuery(getSGDVectorArr());
    }

    private SGDVector[] getSGDVectorArr() {
        SGDVector[] sgdVectors = new SGDVector[vectors.length];
        int n = 0;
        for (Pair<SGDVector,T> vector : vectors) {
            sgdVectors[n] = vector.getA();
            n++;
        }
        return sgdVectors;
    }

    /**
     * It's a specialised non-final pair used for buffering and to reduce object creation.
     * @param <T> The output type.
     */
    private static final class OutputDoublePair<T extends Output<T>> implements Comparable<OutputDoublePair<T>> {
        T output;
        double value;

        public OutputDoublePair(T output, double value) {
            this.output = output;
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            OutputDoublePair<?> that = (OutputDoublePair<?>) o;
            return Double.compare(that.value, value) == 0 &&
                    output.equals(that.output);
        }

        @Override
        public int hashCode() {
            return Objects.hash(output, value);
        }

        @Override
        public int compareTo(OutputDoublePair<T> o) {
            return Double.compare(value, o.value);
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

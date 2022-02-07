/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.neighbour.bruteforce;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.neighbour.NeighboursQuery;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * A naive brute-force nearest neighbour query implementation.
 */
public final class NeighboursBruteForceNaive implements NeighboursQuery {

    private final SGDVector[] data;
    private final DistanceType distanceType;
    private final int numThreads;

    /**
     * Constructs a naive brute-force nearest neighbour query object using the supplied parameters.
     * @param data the data that will be used for neighbour queries.
     * @param distanceType The distance function.
     * @param numThreads The number of threads to be used to parallelize the computation.
     */
    NeighboursBruteForceNaive(SGDVector[] data, DistanceType distanceType, int numThreads) {
        this.data = data;
        this.distanceType = distanceType;
        this.numThreads = numThreads;
    }

    @Override
    public List<Pair<Integer, Double>> query(SGDVector point, int k) {
        MutablePair[] indexDistanceArr = new MutablePair[k];

        for (int i = 0; i < k; i++) {
            indexDistanceArr[i] = new MutablePair(0, Double.MAX_VALUE);
        }

        for (int neighbor = 0; neighbor < data.length; neighbor++) {
            double distance = DistanceType.getDistance(point, data[neighbor], distanceType);

            // Check at which position in the nearest distances the current distance would fit.
            // k is typically small, but if cases with larger values of k become prevalent, this should be replaced
            // with a binary search
            int neighborIndex = k;
            while (neighborIndex >= 1 && distance < indexDistanceArr[neighborIndex - 1].value) {
                neighborIndex--;
            }

            // Shift elements in the array to make room for the current distance
            // The for loop could be written as an arraycopy, but the result is not particularly readable, and
            // numNeighbors is typically quite small
            if (neighborIndex < k) {
                for (int shiftIndex = k - 1; shiftIndex > neighborIndex; shiftIndex--) {
                    indexDistanceArr[shiftIndex] = indexDistanceArr[shiftIndex - 1];
                }
                indexDistanceArr[neighborIndex] = new MutablePair(neighbor, distance);
            }
        }

        List<Pair<Integer, Double>> indexDistancePairList = new ArrayList<>();
        for (MutablePair mutablePair : indexDistanceArr) {
            indexDistancePairList.add(new Pair<>(mutablePair.index, mutablePair.value));
        }

        return indexDistancePairList;
    }

    @Override
    public List<List<Pair<Integer, Double>>> query(SGDVector[] points, int k) {
        int numQueries = points.length;

        @SuppressWarnings("unchecked")
        List<Pair<Integer, Double>>[] indexDistancePairListArray = (List<Pair<Integer, Double>>[]) new ArrayList[numQueries];

        // When the number of threads is 1, the overhead of thread pools must be avoided
        if (numThreads == 1) {
            for (int point = 0; point < numQueries; point++) {
                indexDistancePairListArray[point] = query(points[point], k);
            }
        } else { // This makes the nearest neighbor queries with multiple threads
            ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
            for (int pointInd = 0; pointInd < numQueries; pointInd++) {
                executorService.execute(new SingleQueryRunnable(pointInd, points[pointInd], k, indexDistancePairListArray));
            }
            executorService.shutdown();
            try {
                boolean finished = executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES);
                if (!finished) {
                    throw new RuntimeException("Parallel execution failed");
                }
            } catch (InterruptedException e) {
                throw new RuntimeException("Parallel execution failed", e);
            }
        }
        return new ArrayList<>(Arrays.asList(indexDistancePairListArray));
    }

    @Override
    public List<List<Pair<Integer, Double>>> queryAll(int k) {
        return this.query(this.data, k);
    }

    /**
     * This is a specific mutable pair used for an internal queue to reduce object creation. Note that this class's
     * ordering is not consistent with its equals method. Furthermore, the ordering of this class is the inverse of the
     * natural ordering on doubles.
     */
    private static final class MutablePair implements Comparable<MutablePair> {
        int index;
        double value;

        public MutablePair(int index, double value) {
            this.index = index;
            this.value = value;
        }

        @Override
        public int compareTo(MutablePair o) {
            // Pass the provided value as the first param to give an inverse natural ordering.
            return Double.compare(o.value, this.value);
        }
    }

    /**
     * A Runnable implementation to make a brute-force nearest neighbour query for parallelization of large numbers
     * of queries. To be used with an {@link ExecutorService}
     */
    private final class SingleQueryRunnable implements Runnable {

        final private SGDVector point;
        final private int k;
        final private int index;
        final List<Pair<Integer, Double>>[] indexDistancePairListArray;

        SingleQueryRunnable(int index, SGDVector point, int k, List<Pair<Integer, Double>>[] indexDistancePairListArray) {
            this.point = point;
            this.k = k;
            this.index = index;
            this.indexDistancePairListArray = indexDistancePairListArray;
        }

        @Override
        public void run() {
            indexDistancePairListArray[index] = query(point, k);
        }
    }
}

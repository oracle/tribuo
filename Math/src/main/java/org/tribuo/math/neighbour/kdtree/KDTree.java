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

package org.tribuo.math.neighbour.kdtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.distance.Distance;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.neighbour.NeighboursQuery;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * A k-d tree nearest neighbour query implementation.
 * <p>
 * For the original algorithm see:
 * <pre>
 * J.L. Bentley "Multidimensional Binary Search Trees Used for Associative Searching", Commun. ACM, Vol 18, Sept. 1975, 509–517
 * <a href="https://doi.org/10.1145/361002.361007">Multidimensional Binary Search Trees Used for Associative Searching</a>
 * </pre>
 * <p>
 * Some aspects of this implementation, particularly the logic which finds the median and partitions the points for each
 * dimension, are derived from:
 * <pre>
 * G. Heineman, G. Pollice, and S. Selkow. "Algorithms in a Nutshell" O'Reilly Media, Inc. 2008.
 * <a href="http://oreilly.com/catalog/9780596516246/">Algorithms in a Nutshell</a>
 * </pre>
 */
public final class KDTree implements NeighboursQuery {

    private final SGDVector[] data;
    private final int numThreads;

    private final DimensionNode root;

    /**
     * Constructs a k-d tree nearest neighbour query object using the supplied parameters.
     *
     * @param data The data that will be used for neighbour queries.
     * @param distance The distance function.
     * @param numThreads The number of threads to be used to parallelize queries for multiple points.
     */
    KDTree(SGDVector[] data, Distance distance, int numThreads) {
        this.data = data;
        this.numThreads = numThreads;

        int numDimensions = data[0].size();

        // The original data is copied into this array of intAndVectors. The new array will be rearranged, throughout the
        // process of tree generation, and the original indices need to maintained. The size of each vector is confirmed
        // to be equal.
        IntAndVector[] points = new IntAndVector[data.length];
        for (int i=0; i < data.length; i++) {
            points[i] = new IntAndVector(i, data[i]);
            if (data[i].size() != numDimensions) {
                throw new IllegalArgumentException("All the SGDVectors must be the same size.");
            }
        }

        root = generateTree(0, numDimensions-1, points, 0, data.length - 1, distance);
    }

    @Override
    public List<Pair<Integer, Double>> query(SGDVector point, int k) {
        DistanceIntAndVectorBoundedMinHeap queue = new DistanceIntAndVectorBoundedMinHeap(k);

        // make a fast initialization of the queue starting with points likely close to the target.
        initializeQueue(point, queue);

        // perform a thorough traversal of the tree to find the k nearest neighbours.
        root.nearest(point, queue, false);

        @SuppressWarnings("unchecked")
        Pair<Integer, Double>[] indexDistanceArr = (Pair<Integer, Double>[]) new Pair[k];
        int i = 1;
        // Use an array to put the polled items from the queue into a sorted ascending order, by distance.
        while (!queue.isEmpty()) {
            MutableDistIntAndVectorTuple tuple = queue.poll();
            indexDistanceArr[k - i] = new Pair<>(tuple.intAndVector.idx, tuple.dist);
            i++;
        }
        return Arrays.asList(indexDistanceArr);
    }

    @Override
    public List<List<Pair<Integer, Double>>> query(SGDVector[] points, int k) {
        int numQueries = points.length;

        @SuppressWarnings("unchecked")
        List<Pair<Integer, Double>>[] indexDistancePairListArray = (List<Pair<Integer, Double>>[]) new List[numQueries];

        // When the number of threads is 1, the overhead of thread pools must be avoided
        if (numThreads == 1) {
            for (int point = 0; point < numQueries; point++) {
                indexDistancePairListArray[point] = query(points[point], k);
            }
        } else { // This makes each k-d tree neighbor query in a separate thread
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
        return Arrays.asList(indexDistancePairListArray);
    }

    @Override
    public List<List<Pair<Integer, Double>>> queryAll(int k) {
        return this.query(this.data, k);
    }

    /**
     * Generate a k-d (sub)tree. This method is recursively called to generate the left and right subtrees.
     * @param d The dimension to be considered for this node of the tree.
     * @param maxD The maximum dimension of the tree.
     * @param points The array of points, which are {@link IntAndVector}s of the point's original index and the features.
     * @param left The lower bound index of for this node of the tree.
     * @param right The upper bound index of for this node of the tree.
     * @param distance The distance function.
     * @return The DimensionNode at the root of the (sub)tree being generated.
     */
    private static DimensionNode generateTree(int d, int maxD, IntAndVector[] points, int left, int right,
                                       Distance distance) {
        // Handle the termination cases
        if (right < left) {
            return null;
        }
        if (right == left) {
            return new DimensionNode(d, points[left], distance);
        }

        // Order the points[left, right] based on the dimension d, such that the mth element will be the median.
        // Points before it (with lower index) will be <= median, although not sorted. Points after it
        // (with higher index) will be >= median, again not sorted.
        int median = 1 + (right-left)/2;
        setMedian(points, median, left, right, d);

        // Construct a dimension node using the median point
        DimensionNode medianNode = new DimensionNode(d, points[left+median-1], distance);

        // Increment the dimension, resetting back to 0 when it exceeds maxD
        d++;
        if (d > maxD) {
            d = 0;
        }

        // Generate the below/left and above/right subtrees.
        medianNode.setBelow(generateTree(d, maxD, points, left, left+median-2, distance));
        medianNode.setAbove(generateTree(d, maxD, points, left+median, right, distance));
        return medianNode;
    }

    /**
     * Set the median point for an array of {@link IntAndVector}s based, for a specific dimension, through recursive partitioning
     * ensuring that points before it (with lower index) will be <= median, although not sorted, and points after it
     * (with higher index) will be >= median, again not sorted. The order of the array will almost certainly be changed.
     *
     * @param points The array of {@link IntAndVector}s. NOTE: The order of the array will be changed.
     * @param median The index of the median point.
     * @param left The left-bound to use during this operation.
     * @param right The right-bound to use during this operation.
     * @param dimension The current dimension being processed.
     */
    private static void setMedian(IntAndVector[] points, int median, int left, int right, int dimension) {
        while(true) {
            int initialPivotIndex = getPivotPointIndex(points, left, right, dimension);
            int newPivotIndex = partitionOnIndex(points, left, right, initialPivotIndex, dimension);

            if (left+median-1 == newPivotIndex) {
                // This is the terminating condition.
                return;
            }

            if (left+median-1 < newPivotIndex) {
                // The left side of the pivot is being processed. Update the right bound, and keep the median the same.
                right = newPivotIndex - 1;
            } else {
                // The right side of the pivot is being processed. The median is decreased, and the left bound is moved up.
                median -= (newPivotIndex + 1 - left);
                left = newPivotIndex + 1;
            }
        }
    }

    /**
     * Get the index of a pivot point within the specified range in the array of {@link IntAndVector}s. The operation is performed
     * for a specific dimension of the vectors defined by the comparator.
     *
     * @param points The array of {@link IntAndVector}s.
     * @param left The left-bound to use during this operation.
     * @param right The right-bound to use during this operation.
     * @param dimension The current dimension being processed.
     * @return The index of the pivot.
     */
    private static int getPivotPointIndex(IntAndVector[] points, int left, int right, int dimension) {
        int midIndex = (left + right) / 2;

        int lowIndex = left;

        if (compareByDimension(points[lowIndex], points[midIndex], dimension) >= 0) {
            lowIndex = midIndex;
            midIndex = left;
        }

        // Now, points[lowIndex] < points[midIndex] for the dimension being compared.
        // Check where points[right] fits in, for the dimension being compared.
        if (compareByDimension(points[right], points[lowIndex], dimension) <= 0) {
            // points[right] <= points[lowIndex] < points[midIndex] for the dimension being compared
            return lowIndex;
        } else if (compareByDimension(points[right], points[midIndex], dimension) <= 0) {
            // points[lowIndex] <= points[right] < points[midIndex]
            return right;
        }

        // points[lowIndex] < points[midIndex] < points[right]
        return midIndex;
    }

    /**
     * Partition the array into two parts around the provided array index. Those less than or equal to the value at the
     * provided array index will be on the left. Those greater than the value at the provided array index will fall
     * on the right. The operation is performed for a specific dimension of the vectors defined by the comparator.
     *
     * @param points The array of {@link IntAndVector}s. NOTE: The order of the array will be changed.
     * @param left The left-bound to use during this operation.
     * @param right The right-bound to use during this operation.
     * @param pivotIndex Index around which the partition is being made.
     * @param dimension The current dimension being processed.
     * @return The correct, updated index of the pivot point.
     */
    private static int partitionOnIndex(IntAndVector[] points, int left, int right, int pivotIndex, int dimension) {
        IntAndVector pivot = points[pivotIndex];
        // for now, move the pivot to the end of the array
        swap(points, right, pivotIndex);

        int store = left;
        for (int idx = left; idx < right; idx++) {
            if (compareByDimension(points[idx], pivot, dimension) <= 0) {
                swap (points, idx, store);
                store++;
            }
        }

        // move the pivot to its correct index
        swap(points, right, store);
        return store;
    }

    /**
     * Swap the points at the provided indices.
     *
     * @param points The array of {@link IntAndVector}s.
     * @param ind1 The first index position.
     * @param ind2 The second index position.
     */
    private static void swap(IntAndVector[] points, int ind1, int ind2) {
        if (ind1 == ind2) {
            return;
        }
        IntAndVector tmpPoint = points[ind1];
        points[ind1] = points[ind2];
        points[ind2] = tmpPoint;
    }

    /**
     * Initialize the queue used throughout a query. First the node which might be the target point's parent is
     * approximated, then the queue is seeded by checking for neighbours from this node.
     * @param point The target point.
     * @param queue The priority queue used to maintain the k nearest neighbours.
     */
    private void initializeQueue(SGDVector point, DistanceIntAndVectorBoundedMinHeap queue) {
        DimensionNode parentOfPoint = approximateParentNode(point);
        parentOfPoint.nearest(point, queue,true);
    }

    /**
     * This makes a fast approximation of the provided point's parent node, if it were being inserted
     * into the tree.
     *
     * @param point The target point.
     * @return The approximate parent node.
     */
    public DimensionNode approximateParentNode(SGDVector point) {
        DimensionNode node = root;
        // A non-leaf node is more desirable for this approximation. A reference to the last node where both child nodes
        // are not null is maintained, in case a leaf node is approximated to be the parent.
        DimensionNode bestNode = node;
        DimensionNode next;
        while (node != null) {
            if (node.getBelow() != null && node.getAbove() != null) {
                bestNode = node;
            }
            // If this point's specific feature is below the node's value at this dimension, search that direction
            if (node.isBelow(point)) {
                next = node.getBelow();
            } else {
                next = node.getAbove();
            }
            if (next == null) {
                // This is the bottom of the tree, the terminating condition.
                break;
            } else {
                node = next;
            }
        }
        return bestNode;
    }


    /**
     * Tuple of index and position.
     */
    static final class IntAndVector {
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
     * This is used to compare two {@link IntAndVector}s by a specific dimension.
     *
     * @param intAndVector1 The first {@link IntAndVector} to use in the comparison.
     * @param intAndVector2 The second {@link IntAndVector} to use in the comparison.
     * @param dimension The current dimension being processed.
     * @return The result of the comparison. See {@link Double#compare}.
     */
    private static int compareByDimension(IntAndVector intAndVector1, IntAndVector intAndVector2, int dimension) {
        return Double.compare(intAndVector1.vector.get(dimension), intAndVector2.vector.get(dimension));
    }

    /**
     * This is a specific mutable tuple object used for an internal queue to reduce object creation. Note that this
     * class's ordering is not consistent with its equals method. Furthermore, the ordering of this class is the
     * inverse of the natural ordering on doubles.
     */
    static final class MutableDistIntAndVectorTuple implements Comparable<MutableDistIntAndVectorTuple> {
        double dist;
        IntAndVector intAndVector;

        public MutableDistIntAndVectorTuple(double dist, IntAndVector intAndVector) {
            this.dist = dist;
            this.intAndVector = intAndVector;
        }

        @Override
        public int compareTo(MutableDistIntAndVectorTuple o) {
            // Pass the provided value as the first param to give an inverse natural ordering.
            return Double.compare(o.dist, this.dist);
        }
    }

    /**
     * A bounded min heap implementation which wraps a priority queue specific to {@link MutableDistIntAndVectorTuple}
     * objects. This logic needs to be encapsulated to handle duplicates.
     */
    static final class DistanceIntAndVectorBoundedMinHeap {
        // A set containing the ids needs to be maintained to prevent duplicates from being added into the
        // queue.
        private final HashSet<Integer> ids = new HashSet<>();
        final int size;
        private final PriorityQueue<MutableDistIntAndVectorTuple> queue;

        DistanceIntAndVectorBoundedMinHeap(int size) {
            this.size = size;
            queue = new PriorityQueue<>(size);
        }

        void boundedOffer(IntAndVector intAndVector, double distance) {
            // Prevent duplicates
            if (ids.contains(intAndVector.idx)) {
                return;
            }

            if (queue.size() < size) {
                queue.offer(new MutableDistIntAndVectorTuple(distance, intAndVector));
                ids.add(intAndVector.idx);
            } else if (Double.compare(distance, queue.peek().dist) < 0) {
                // remove the intAndVector from the queue, and its id from the set.
                MutableDistIntAndVectorTuple tuple = this.poll();

                tuple.dist = distance;
                tuple.intAndVector = intAndVector;
                queue.offer(tuple);
                ids.add(intAndVector.idx);
            }
        }

        MutableDistIntAndVectorTuple peek() {
            return queue.peek();
        }

        MutableDistIntAndVectorTuple poll() {
            MutableDistIntAndVectorTuple tuple = queue.poll();
            ids.remove(tuple.intAndVector.idx);
            return tuple;
        }

        boolean isFull() {
            return queue.size() == size;
        }

        boolean isEmpty() {
            return queue.isEmpty();
        }
    }

    /**
     * A Runnable implementation to make a k-d tree nearest neighbour query for parallelization of large numbers
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

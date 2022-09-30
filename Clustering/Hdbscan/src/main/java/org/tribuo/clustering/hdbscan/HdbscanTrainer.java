/*
 * Copyright (c) 2021-2022, Oracle and/or its affiliates. All rights reserved.
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
package org.tribuo.clustering.hdbscan;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ImmutableClusteringInfo;
import org.tribuo.clustering.hdbscan.protos.ClusterExemplarProto;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.neighbour.NeighboursQuery;
import org.tribuo.math.neighbour.NeighboursQueryFactory;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.io.Serializable;
import java.time.OffsetDateTime;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * An HDBSCAN* trainer which generates a hierarchical, density-based clustering representation
 * of the supplied data.
 * <p>
 * The cluster assignments and outlier scores can be retrieved from the model after training. Outliers or noise
 * points are assigned the label 0.
 * <p>
 * For the HDBSCAN* algorithm see:
 * <pre>
 * R.J.G.B. Campello, D. Moulavi, A. Zimek and J. Sander "Hierarchical Density Estimates for Data Clustering,
 * Visualization, and Outlier Detection", ACM Trans. on Knowledge Discovery from Data, Vol 10, 1 (July 2015), 1-51.
 * <a href="http://lapad-web.icmc.usp.br/?portfolio_1=a-handful-of-experiments">HDBSCAN*</a>
 * </pre>
 * <p>
 * For this specific implementation and prediction technique, see:
 * <pre>
 * G. Stewart, M. Al-Khassaweneh. "An Implementation of the HDBSCAN* Clustering Algorithm",
 * Applied Sciences. 2022; 12(5):2405.
 * <a href="https://doi.org/10.3390/app12052405">Manuscript</a>
 * </pre>
 */
public final class HdbscanTrainer implements Trainer<ClusterID> {
    private static final Logger logger = Logger.getLogger(HdbscanTrainer.class.getName());

    static final int OUTLIER_NOISE_CLUSTER_LABEL = 0;

    private static final double MAX_OUTLIER_SCORE = 0.9999;

    /**
     * Available distance functions.
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

    @Config(mandatory = true, description = "The minimum number of points required to form a cluster.")
    private int minClusterSize;

    @Deprecated
    @Config(description = "The distance function to use. This is now deprecated.")
    private Distance distanceType;

    @Config(description = "The distance function to use.")
    private org.tribuo.math.distance.Distance dist;

    @Config(mandatory = true, description = "The number of nearest-neighbors to use in the initial density approximation. " +
        "This includes the point itself.")
    private int k;

    @Deprecated
    @Config(description = "The number of threads to use for training. This is now deprecated since it is a field on the " +
        "NeighboursQueryFactory object.")
    private int numThreads = 1;

    @Config(description = "The nearest neighbour implementation factory to use.")
    private NeighboursQueryFactory neighboursQueryFactory;

    private int trainInvocationCounter;

    /**
     * for olcut.
     */
    private HdbscanTrainer() {}

    /**
     * Constructs an HDBSCAN* trainer with only the minClusterSize parameter.
     *
     * @param minClusterSize The minimum number of points required to form a cluster.
     * {@link #dist} defaults to {@link DistanceType#L2}, {@link #k} defaults to {@link #minClusterSize},
     * {@link #numThreads} defaults to 1 and {@link #neighboursQueryFactory} defaults to
     * {@link NeighboursBruteForceFactory}.
     */
    public HdbscanTrainer(int minClusterSize) {
        this(minClusterSize, DistanceType.L2.getDistance(), minClusterSize, 1, NeighboursQueryFactoryType.BRUTE_FORCE);
    }

    /**
     * Constructs an HDBSCAN* trainer using the supplied parameters. {@link #neighboursQueryFactory} defaults to
     * {@link NeighboursBruteForceFactory}.
     * @deprecated
     * This Constructor is deprecated in version 4.3.
     *
     * @param minClusterSize The minimum number of points required to form a cluster.
     * @param distanceType The distance function.
     * @param k The number of nearest-neighbors to use in the initial density approximation.
     * @param numThreads The number of threads.
     */
    @Deprecated
    public HdbscanTrainer(int minClusterSize, Distance distanceType, int k, int numThreads) {
        this(minClusterSize, distanceType.getDistanceType().getDistance(), k, numThreads, NeighboursQueryFactoryType.BRUTE_FORCE);
    }

    /**
     * Constructs an HDBSCAN* trainer using the supplied parameters.
     *
     * @param minClusterSize The minimum number of points required to form a cluster.
     * @param dist The distance function.
     * @param k The number of nearest-neighbors to use in the initial density approximation.
     * @param numThreads The number of threads.
     * @param nqFactoryType The nearest neighbour query implementation factory to use.
     */
    public HdbscanTrainer(int minClusterSize, org.tribuo.math.distance.Distance dist, int k, int numThreads, NeighboursQueryFactoryType nqFactoryType) {
        this.minClusterSize = minClusterSize;
        this.dist = dist;
        this.k = k;
        this.numThreads = numThreads;
        this.neighboursQueryFactory = NeighboursQueryFactoryType.getNeighboursQueryFactory(nqFactoryType, dist, numThreads);
    }

    /**
     * Constructs an HDBSCAN* trainer using the supplied parameters.
     *
     * @param minClusterSize The minimum number of points required to form a cluster.
     * @param k The number of nearest-neighbors to use in the initial density approximation.
     * @param neighboursQueryFactory The nearest neighbour query implementation factory to use.
     */
    public HdbscanTrainer(int minClusterSize, int k, NeighboursQueryFactory neighboursQueryFactory) {
        this.minClusterSize = minClusterSize;
        this.dist = neighboursQueryFactory.getDistance();
        this.k = k;
        this.neighboursQueryFactory = neighboursQueryFactory;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        if (this.distanceType != null) {
            if (this.dist != null) {
                throw new PropertyException("distType", "Both distType and distanceType must not both be set.");
            } else {
                this.dist = this.distanceType.getDistanceType().getDistance();
                this.distanceType = null;
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
    public HdbscanModel train(Dataset<ClusterID> examples, Map<String, Provenance> runProvenance) {
        // increment the invocation count.
        TrainerProvenance trainerProvenance;
        synchronized (this) {
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();

        SGDVector[] data = new SGDVector[examples.size()];
        int n = 0;
        for (Example<ClusterID> example : examples) {
            if (example.size() == featureMap.size()) {
                data[n] = DenseVector.createDenseVector(example, featureMap, false);
            } else {
                data[n] = SparseVector.createSparseVector(example, featureMap, false);
            }
            n++;
        }

        DenseVector coreDistances = calculateCoreDistances(data, k, neighboursQueryFactory);
        ExtendedMinimumSpanningTree emst = constructEMST(data, coreDistances, dist);

        double[] pointNoiseLevels = new double[data.length];    // The levels at which each point becomes noise
        int[] pointLastClusters = new int[data.length];         // The last label of each point before becoming noise
        Map<Integer, int[]> hierarchy = new HashMap<>();        // The HDBSCAN* hierarchy
        List<HdbscanCluster> clusters = computeHierarchyAndClusterTree(emst, minClusterSize, pointNoiseLevels,
                                                                       pointLastClusters, hierarchy);
        propagateTree(clusters);
        List<Integer> clusterLabels = findProminentClusters(hierarchy, clusters, data.length);
        DenseVector outlierScoresVector = calculateOutlierScores(pointNoiseLevels, pointLastClusters, clusters);
        Map<Integer, List<Pair<Double, Integer>>> clusterAssignments =  generateClusterAssignments(clusterLabels, outlierScoresVector);

        // Use the cluster assignments to establish the clustering info
        Map<Integer, MutableLong> counts = new HashMap<>();
        for (Entry<Integer, List<Pair<Double, Integer>>> e : clusterAssignments.entrySet()) {
            counts.put(e.getKey(), new MutableLong(e.getValue().size()));
        }
        ImmutableOutputInfo<ClusterID> outputMap = new ImmutableClusteringInfo(counts);

        // Compute the cluster exemplars.
        List<ClusterExemplar> clusterExemplars = computeExemplars(data, clusterAssignments, dist);

        // Get the outlier score value for points that are predicted as noise points.
        double noisePointsOutlierScore = getNoisePointsOutlierScore(clusterAssignments);

        logger.log(Level.INFO, "Hdbscan is done.");

        ModelProvenance provenance = new ModelProvenance(HdbscanModel.class.getName(), OffsetDateTime.now(),
                examples.getProvenance(), trainerProvenance, runProvenance);

        return new HdbscanModel("hdbscan-model", provenance, featureMap, outputMap, clusterLabels, outlierScoresVector,
                                clusterExemplars, dist, noisePointsOutlierScore);
    }

    @Override
    public HdbscanModel train(Dataset<ClusterID> dataset) {
        return train(dataset, Collections.emptyMap());
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public void setInvocationCount(int newInvocationCount) {
        if(newInvocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        } else {
            trainInvocationCounter = newInvocationCount;
        }
    }

    /**
     * Calculates the core distance for every point in the data set.
     *
     * @param data An array of {@link DenseVector} containing the data.
     * @param k The number of nearest-neighbors to use in these calculations.
     * @param neighboursQueryFactory The nearest neighbour implementation factory.
     * @return A {@link DenseVector} containing the core distances for every point.
     */
    private static DenseVector calculateCoreDistances(SGDVector[] data, int k, NeighboursQueryFactory neighboursQueryFactory) {
        DenseVector coreDistances = new DenseVector(data.length);

        // A value of k=1 will not return any neighbouring points
        if (k == 1) {
            return coreDistances;
        }

        NeighboursQuery nq = neighboursQueryFactory.createNeighboursQuery(data);
        List<List<Pair<Integer, Double>>> indexDistancePairListOfLists = nq.queryAll(k);
        for (int point = 0; point < data.length; point++) {
            coreDistances.set(point, indexDistancePairListOfLists.get(point).get(k-1).getB());
        }

        return coreDistances;
    }

    /**
     * Constructs an extended minimum spanning tree of mutual reachability distances from the data, given the
     * core distances for each point.
     * @param data An array of {@link DenseVector} containing the data.
     * @param coreDistances A {@link DenseVector} containing the core distances for every point.
     * @param dist The distance metric to employ.
     * @return An {@link ExtendedMinimumSpanningTree} representation of the data using the mutual reachability distances,
     * and the graph is sorted by edge weight in ascending order.
     */
    private static ExtendedMinimumSpanningTree constructEMST(SGDVector[] data, DenseVector coreDistances,
                                                             org.tribuo.math.distance.Distance dist) {
        // One bit is set (true) for each attached point, and unset (false) for unattached points:
        BitSet attachedPoints = new BitSet(data.length);

        // Each point has a current neighbor point in the tree, and a corresponding current neighbor's distance
        int[] nearestMRDNeighbors = new int[2*data.length - 1];
        double[] nearestMRDDistances = new double[2*data.length - 1];

        for (int i = 0; i < data.length-1; i++) {
            nearestMRDDistances[i] = Double.MAX_VALUE;
        }

        // The MST is expanded starting with the last point in the data set:
        int currentPoint = data.length-1;
        int numAttachedPoints = 1;
        attachedPoints.set(data.length-1);

        // Continue attaching points to the MST until all points are attached:
        while (numAttachedPoints < data.length) {
            int nearestMRDPoint = -1;
            double nearestMRDDistance = Double.MAX_VALUE;

            // Iterate through all unattached points, updating distances using the current point:
            for (int neighbor = 0; neighbor < data.length; neighbor++) {
                if (currentPoint == neighbor || attachedPoints.get(neighbor)) {
                    continue;
                }

                double mutualReachabilityDistance = dist.computeDistance(data[currentPoint], data[neighbor]);
                if (coreDistances.get(currentPoint) > mutualReachabilityDistance) {
                    mutualReachabilityDistance = coreDistances.get(currentPoint);
                }
                if (coreDistances.get(neighbor) > mutualReachabilityDistance) {
                    mutualReachabilityDistance = coreDistances.get(neighbor);
                }
                if (mutualReachabilityDistance < nearestMRDDistances[neighbor]) {
                    nearestMRDDistances[neighbor] = mutualReachabilityDistance;
                    nearestMRDNeighbors[neighbor] = currentPoint;
                }

                // Check if the unattached point being updated is the closest to the tree:
                if (nearestMRDDistances[neighbor] <= nearestMRDDistance) {
                    nearestMRDDistance = nearestMRDDistances[neighbor];
                    nearestMRDPoint = neighbor;
                }
            }

            //Attach the closest point found in this iteration to the tree:
            attachedPoints.set(nearestMRDPoint);
            numAttachedPoints++;
            currentPoint = nearestMRDPoint;
        }

        // Create an array for vertices in the tree that each point attached to:
        int[] otherVertexIndices = new int[2*data.length - 1];
        for (int i = 0; i < data.length-1; i++) {
            otherVertexIndices[i] = i;
        }

        // Attach self edges:
        for (int i = data.length-1; i < data.length*2-1; i++) {
            int vertex = i - (data.length-1);
            nearestMRDNeighbors[i] = vertex;
            otherVertexIndices[i] = vertex;
            nearestMRDDistances[i] = coreDistances.get(vertex);
        }
        ExtendedMinimumSpanningTree emst = new ExtendedMinimumSpanningTree(data.length, nearestMRDNeighbors,
                                                                           otherVertexIndices, nearestMRDDistances);
        return emst;
    }

    /**
     * Computes the hierarchy and cluster tree from the extended minimum spanning tree. Additionally, the level
     * at which each point becomes noise is recorded.
     * @param emst The extended minimum spanning tree which has been sorted by edge weight in ascending order.
     * @param minClusterSize The minimum number of points which a cluster needs to be a valid cluster
     * @param pointNoiseLevels A double[] to be populated with the edge weight levels at which each point becomes noise
     * @param pointLastClusters An int[] to be populated with the last label of each point had before becoming noise
     * @param hierarchy The HDBSCAN* hierarchy.
     * @return The cluster tree.
     */
    private static List<HdbscanCluster> computeHierarchyAndClusterTree(ExtendedMinimumSpanningTree emst, int minClusterSize,
                                                                    double[] pointNoiseLevels, int[] pointLastClusters,
                                                                    Map<Integer, int[]> hierarchy) {

        int lineCount = 0; //Indicates the level number in the hierarchy

        // The current edge being removed from the MST:
        int currentEdgeIndex = emst.getNumEdges()-1;

        int nextClusterLabel = 2;    // all points are currently assigned to cluster 1
        boolean nextLevelSignificant = true;

        // The current cluster numbers of each point in the data set:
        int[] currentClusterLabels = new int[emst.getNumVertices()];
        Arrays.fill(currentClusterLabels, 1);

        // A list of clusters in the cluster tree, with the 0th cluster (noise) null:
        ArrayList<HdbscanCluster> clusters = new ArrayList<>();
        clusters.add(HdbscanCluster.NOT_A_CLUSTER);
        clusters.add(new HdbscanCluster(1, HdbscanCluster.NOT_A_CLUSTER, Double.NaN, emst.getNumVertices()));

        // Sets for the clusters and vertices that are affected by the edge(s) being removed. The labels depend
        // on being ordered.
        TreeSet<Integer> affectedClusterLabels = new TreeSet<>();
        // There is no ordering requirement on the vertices
        HashSet<Integer> affectedVertices = new HashSet<>();

        while(currentEdgeIndex >= 0) {
            double currentEdgeWeight = emst.getEdgeWeightAtIndex(currentEdgeIndex);
            ArrayList<HdbscanCluster> newClusters = new ArrayList<>();

            // Remove all edges tied with the current edge weight, and store relevant clusters and vertices:
            while (currentEdgeIndex >= 0 && emst.getEdgeWeightAtIndex(currentEdgeIndex) == currentEdgeWeight){
                int firstVertex = emst.getFirstVertexAtIndex(currentEdgeIndex);
                int secondVertex = emst.getSecondVertexAtIndex(currentEdgeIndex);
                // The cast of these variables to Integer is required to get the correct overload of the remove method.
                emst.getEdgeListForVertex(firstVertex).remove((Integer)secondVertex);
                emst.getEdgeListForVertex(secondVertex).remove((Integer)firstVertex);

                if (currentClusterLabels[firstVertex] != 0) {
                    affectedVertices.add(firstVertex);
                    affectedVertices.add(secondVertex);
                    affectedClusterLabels.add(currentClusterLabels[firstVertex]);
                }

                currentEdgeIndex--;
            }

            if (affectedClusterLabels.isEmpty()) {
                continue;
            }

            // Check each cluster affected for a possible split:
            while (!affectedClusterLabels.isEmpty()) {
                int examinedClusterLabel = affectedClusterLabels.last();
                affectedClusterLabels.remove(examinedClusterLabel);
                TreeSet<Integer> examinedVertices = new TreeSet<>();

                // Get all affected vertices that are members of the cluster currently being examined:
                Iterator<Integer> vertexIterator = affectedVertices.iterator();
                while (vertexIterator.hasNext()) {
                    int vertex = vertexIterator.next();

                    if (currentClusterLabels[vertex] == examinedClusterLabel) {
                        examinedVertices.add(vertex);
                        vertexIterator.remove();
                    }
                }

                TreeSet<Integer> firstChildCluster = null;
                ArrayDeque<Integer> unexploredFirstChildClusterPoints = null;
                int numChildClusters = 0;

                /*
                 * Check if the cluster has split or shrunk by exploring the graph from each affected
                 * vertex.  If there are two or more valid child clusters (each has >= minClusterSize
                 * points), the cluster has split.
                 * Note that firstChildCluster will only be fully explored if there is a cluster
                 * split, otherwise, only spurious components are fully explored, in order to label
                 * them noise.
                 */
                while (!examinedVertices.isEmpty()) {
                    TreeSet<Integer> constructingSubCluster = new TreeSet<>();
                    ArrayDeque<Integer> unexploredSubClusterPoints = new ArrayDeque<>();
                    boolean anyEdges = false;
                    boolean incrementedChildCount = false;

                    int rootVertex = examinedVertices.last();
                    constructingSubCluster.add(rootVertex);
                    unexploredSubClusterPoints.add(rootVertex);
                    examinedVertices.remove(rootVertex);

                    // Explore this potential child cluster as long as there are unexplored points:
                    while (!unexploredSubClusterPoints.isEmpty()) {
                        int vertexToExplore = unexploredSubClusterPoints.poll();

                        for (int neighbor : emst.getEdgeListForVertex(vertexToExplore)) {
                            anyEdges = true;
                            if (constructingSubCluster.add(neighbor)) {
                                unexploredSubClusterPoints.add(neighbor);
                                examinedVertices.remove(neighbor);
                            }
                        }

                        // Check if this potential child cluster is a valid cluster:
                        if (!incrementedChildCount && constructingSubCluster.size() >= minClusterSize && anyEdges) {
                            incrementedChildCount = true;
                            numChildClusters++;

                            // If this is the first valid child cluster, stop exploring it:
                            if (firstChildCluster == null) {
                                firstChildCluster = constructingSubCluster;
                                unexploredFirstChildClusterPoints = unexploredSubClusterPoints;
                                break;
                            }
                        }
                    }

                    // If there could be a split, and this child cluster is valid:
                    if (numChildClusters >= 2 && constructingSubCluster.size() >= minClusterSize && anyEdges) {

                        // Check this child cluster is not equal to the unexplored first child cluster:
                        int firstChildClusterMember = firstChildCluster.last();
                        if (constructingSubCluster.contains(firstChildClusterMember)) {
                            numChildClusters--;
                        } else {   // Otherwise, create a new cluster:
                            HdbscanCluster parentCluster = clusters.get(examinedClusterLabel);
                            HdbscanCluster newCluster = parentCluster.createNewCluster(constructingSubCluster, currentClusterLabels,
                                nextClusterLabel, currentEdgeWeight);
                            newClusters.add(newCluster);
                            clusters.add(newCluster);
                            nextClusterLabel++;
                        }
                    } else if (constructingSubCluster.size() < minClusterSize || !anyEdges){
                        // If this child cluster is not valid cluster, assign it to noise:
                        HdbscanCluster parentCluster = clusters.get(examinedClusterLabel);
                        parentCluster.createNewCluster(constructingSubCluster, currentClusterLabels, OUTLIER_NOISE_CLUSTER_LABEL, currentEdgeWeight);

                        for (int point : constructingSubCluster) {
                            pointNoiseLevels[point] = currentEdgeWeight;
                            pointLastClusters[point] = examinedClusterLabel;
                        }
                    }
                }

                // Finish exploring and cluster the first child cluster if there was a split and it was not already clustered:
                if (numChildClusters >= 2 && currentClusterLabels[firstChildCluster.first()] == examinedClusterLabel) {

                    while (!unexploredFirstChildClusterPoints.isEmpty()) {
                        int vertexToExplore = unexploredFirstChildClusterPoints.poll();

                        for (int neighbor : emst.getEdgeListForVertex(vertexToExplore)) {
                            if (firstChildCluster.add(neighbor)) {
                                unexploredFirstChildClusterPoints.add(neighbor);
                            }
                        }
                    }

                    HdbscanCluster parentCluster = clusters.get(examinedClusterLabel);
                    HdbscanCluster newCluster = parentCluster.createNewCluster(firstChildCluster, currentClusterLabels,
                                                                               nextClusterLabel, currentEdgeWeight);
                    newClusters.add(newCluster);
                    clusters.add(newCluster);
                    nextClusterLabel++;
                }
            }

            // increment the current level of the hierarchy:
            if (nextLevelSignificant || !newClusters.isEmpty()) {
                // this is used for below to set the hierarchy level
                lineCount++;
            }

            for (HdbscanCluster newCluster : newClusters) {
                int[] hierarchyLevelLabels = new int[emst.getNumVertices()];
                System.arraycopy(currentClusterLabels, 0, hierarchyLevelLabels, 0, currentClusterLabels.length);
                newCluster.setHierarchyLevel(lineCount);
                hierarchy.put(lineCount, hierarchyLevelLabels);
            }

            if (newClusters.isEmpty()) {
                nextLevelSignificant = false;
            } else {
                nextLevelSignificant = true;
            }
        }
        return clusters;
    }

    /**
     * Propagates stability and lowest child death level from each child cluster to its parent cluster in the tree.
     * This step is required before calling prominent clusters can be identified or outlier scores can be calculated.
     * @param clusters A list of clusters representing the cluster tree.
     */
    private static void propagateTree(List<HdbscanCluster> clusters) {
        PriorityQueue<HdbscanCluster> clustersToExamine = new PriorityQueue<>();
        BitSet addedToExaminationList = new BitSet(clusters.size());

        // Find all leaf clusters in the cluster tree:
        for (HdbscanCluster cluster : clusters) {
            if (cluster != HdbscanCluster.NOT_A_CLUSTER && !cluster.hasChildren()) {
                clustersToExamine.add(cluster);
                addedToExaminationList.set(cluster.getLabel());
            }
        }

        // Iterate through every cluster, propagating stability from children to parents:
        while (!clustersToExamine.isEmpty()) {
            HdbscanCluster currentCluster = clustersToExamine.poll();
            currentCluster.propagate();

            if (currentCluster.getParent() != HdbscanCluster.NOT_A_CLUSTER) {
                HdbscanCluster parent = currentCluster.getParent();

                if (!addedToExaminationList.get(parent.getLabel())) {
                    clustersToExamine.add(parent);
                    addedToExaminationList.set(parent.getLabel());
                }
            }
        }
    }

    /**
     * Produces the cluster labels using cluster stability values.
     *
     * @param hierarchy The HDBSCAN* hierarchy.
     * @param clusters A list of clusters representing the cluster tree.
     * @param numPoints The number of points in the dataset.
     * @return A list containing every point's cluster label.
     */
    private static List<Integer> findProminentClusters(Map<Integer, int[]> hierarchy, List<HdbscanCluster> clusters,
                                                      int numPoints) {

        // Take the list of propagated clusters from the root cluster:
        List<HdbscanCluster> solution = clusters.get(1).getPropagatedDescendants();

        List<Integer> clusterLabels = new ArrayList<>(Collections.nCopies(numPoints, OUTLIER_NOISE_CLUSTER_LABEL));

        // Store all the levels at which to find the birth points for the flat clustering:
        TreeMap<Integer, List<Integer>> significantLevels = new TreeMap<>();
        for (HdbscanCluster cluster: solution) {
            List<Integer> clusterList = significantLevels.computeIfAbsent(cluster.getHierarchyLevel(), p -> new ArrayList<>());

            clusterList.add(cluster.getLabel());
        }

        // Go through the hierarchy, setting labels for the flat clustering:
        while (!significantLevels.isEmpty()) {
            Map.Entry<Integer, List<Integer>> entry = significantLevels.pollFirstEntry();
            List<Integer> clusterList = entry.getValue();
            Integer hierarchyLevel = entry.getKey();

            int[] hierarchyLevelLabels = hierarchy.get(hierarchyLevel);

            for (int i = 0; i < hierarchyLevelLabels.length; i++) {
                int label = hierarchyLevelLabels[i];
                if (clusterList.contains(label)) {
                    clusterLabels.set(i, label);
                }
            }
        }
        return Collections.unmodifiableList(clusterLabels);
    }

    /**
     * Produces the outlier score for each point in the data.
     *
     * @param clusters A list of Clusters forming a cluster tree which has already been propagated.
     * @param pointNoiseLevels A double[] with the levels at which each point becomes noise.
     * @param pointLastClusters An int[] with the last label of each point before becoming noise.
     * @return A {@link DenseVector} containing the outlier scores.
     */
    private static DenseVector calculateOutlierScores(double[] pointNoiseLevels, int[] pointLastClusters,
                                                                 List<HdbscanCluster> clusters) {

        int numPoints = pointNoiseLevels.length;
        DenseVector outlierScores = new DenseVector(numPoints);

        // Iterate through each point, calculating its outlier score:
        for (int i = 0; i < numPoints; i++) {
            double epsilonMax = clusters.get(pointLastClusters[i]).getPropagatedLowestChildSplitLevel();
            double epsilon = pointNoiseLevels[i];

            double score = 0;
            if (epsilon != 0) {
                score = 1 - (epsilonMax / epsilon);
            }

            outlierScores.set(i, score);
        }
        return outlierScores;
    }

    /**
     * Generate a map keyed by the cluster label and the data points assigned to them. The representation of a data
     * points are their index in the dataset, and their outlier score. In this case, the data points are being sorted
     * by their outlier scores, as convenience to quickly establish the exemplars from the dataset.
     *
     * @param clusterLabels A list containing every point's cluster label.
     * @param outlierScoresVector A {@link DenseVector} containing the outlier scores.
     * @return A map of the cluster labels, and the points assigned to them.
     */
    private static Map<Integer, List<Pair<Double, Integer>>> generateClusterAssignments(List<Integer> clusterLabels,
                                                                           DenseVector outlierScoresVector) {
        // A map value of TreeMap<Double, Integer>> provides a sorting of the entries by outlier score.
        Map<Integer, List<Pair<Double, Integer>>> clusterAssignments = new HashMap<>();
        for (int i = 0; i < clusterLabels.size(); i++) {
            Integer clusterLabel = clusterLabels.get(i);
            Double outlierScore = outlierScoresVector.get(i);
            List<Pair<Double, Integer>> outlierScoreIndexList = clusterAssignments.computeIfAbsent(clusterLabel, j -> new ArrayList<>());
            outlierScoreIndexList.add(new Pair<>(outlierScore, i));
        }
        return clusterAssignments;
    }

    /**
     * Compute the exemplars. These are representative points which are subsets of their clusters, and
     * will be used for prediction on unseen data points.
     *
     * @param data An array of {@link DenseVector} containing the data.
     * @param clusterAssignments A map of the cluster labels, and the points assigned to them.
     * @param dist The distance metric to employ.
     * @return A list of {@link ClusterExemplar}s which are used for predictions.
     */
    private static List<ClusterExemplar> computeExemplars(SGDVector[] data, Map<Integer, List<Pair<Double, Integer>>> clusterAssignments,
                                                          org.tribuo.math.distance.Distance dist) {
        List<ClusterExemplar> clusterExemplars = new ArrayList<>();
        // The formula to calculate the exemplar number. This calculates the number of exemplars to be used for this
        // configuration. The appropriate number of exemplars is important for prediction. At the time, this
        // provides a good value for most scenarios.
        int numExemplars = (int) Math.sqrt(data.length / 2.0) + clusterAssignments.size();

        for (Entry<Integer, List<Pair<Double, Integer>>> e : clusterAssignments.entrySet()) {
            int clusterLabel = e.getKey();

            if (clusterLabel != OUTLIER_NOISE_CLUSTER_LABEL) {
                List<Pair<Double, Integer>> outlierScoreIndexList = clusterAssignments.get(clusterLabel);

                // Put the items into a TreeMap. This achieves the required sorting and removes duplicate outlier scores
                // to provide the best samples.
                TreeMap<Double, Integer> outlierScoreIndexTree = new TreeMap<>();
                outlierScoreIndexList.forEach(p -> outlierScoreIndexTree.put(p.getA(), p.getB()));
                int numExemplarsThisCluster = e.getValue().size() * numExemplars / data.length;
                if (numExemplarsThisCluster == 0) {
                    numExemplarsThisCluster = 1;
                }
                else if (numExemplarsThisCluster > outlierScoreIndexTree.size()) {
                    numExemplarsThisCluster = outlierScoreIndexTree.size();
                }

                // First, get the entries that will be used for cluster exemplars.
                // The first node is polled from the tree, which has the lowest outlier score out of the remaining
                // points assigned this cluster.
                List<Entry<Double, Integer>> partialClusterExemplars = new ArrayList<>();
                Stream<Integer> intStream = IntStream.range(0, numExemplarsThisCluster).boxed();
                intStream.forEach((i) -> partialClusterExemplars.add(outlierScoreIndexTree.pollFirstEntry()));

                // For each of the partial exemplars in this cluster, iterate the remaining nodes in the tree to find
                // the maximum distance between the exemplar and the members of the cluster. The other exemplars don't
                // need to be checked here since they won't be on the fringe of the cluster.
                for (Entry<Double, Integer> partialClusterExemplar : partialClusterExemplars) {
                    SGDVector features = data[partialClusterExemplar.getValue()];
                    double maxInnerDist = Double.NEGATIVE_INFINITY;
                    for (Entry<Double, Integer> entry : outlierScoreIndexTree.entrySet()) {
                        double distance = dist.computeDistance(features, data[entry.getValue()]);
                        if (distance > maxInnerDist){
                            maxInnerDist = distance;
                        }
                    }
                    clusterExemplars.add(new ClusterExemplar(clusterLabel, partialClusterExemplar.getKey(), features,
                                                             maxInnerDist));
                }
            }
        }
        return clusterExemplars;
    }

    /**
     * Determine the outlier score value for points that are predicted as noise points.
     *
     * @param clusterAssignments A map of the cluster labels, and the points assigned to them.
     * @return An outlier score value for points predicted as noise points.
     */
    private static double getNoisePointsOutlierScore(Map<Integer, List<Pair<Double, Integer>>> clusterAssignments) {

        List<Pair<Double, Integer>> outlierScoreIndexList = clusterAssignments.get(OUTLIER_NOISE_CLUSTER_LABEL);
        if ((outlierScoreIndexList == null) || outlierScoreIndexList.isEmpty()) {
            return MAX_OUTLIER_SCORE;
        }

        double upperOutlierScoreBound = Double.NEGATIVE_INFINITY;
        for (Pair<Double, Integer> outlierScoreIndex : outlierScoreIndexList) {
            if (outlierScoreIndex.getA() > upperOutlierScoreBound) {
                upperOutlierScoreBound = outlierScoreIndex.getA();
            }
        }
        return upperOutlierScoreBound;
    }

    @Override
    public String toString() {
        return "HdbscanTrainer(minClusterSize=" + minClusterSize + ",distanceType=" + dist + ",k=" + k + ",numThreads=" + numThreads + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    /**
     * A cluster exemplar, with attributes for the point's label, outlier score and its features.
     */
    public final static class ClusterExemplar implements Serializable {
        private static final long serialVersionUID = 1L;

        private final Integer label;
        private final Double outlierScore;
        private final SGDVector features;
        private final Double maxDistToEdge;

        ClusterExemplar(Integer label, Double outlierScore, SGDVector features, Double maxDistToEdge) {
            this.label = label;
            this.outlierScore = outlierScore;
            this.features = features;
            this.maxDistToEdge = maxDistToEdge;
        }

        /**
         * Get the label in this exemplar.
         * @return The label.
         */
        public Integer getLabel() {
            return label;
        }

        /**
         * Get the outlier score in this exemplar.
         * @return The outlier score.
         */
        public Double getOutlierScore() {
            return outlierScore;
        }

        /**
         * Get the feature vector in this exemplar.
         * @return The feature vector.
         */
        public SGDVector getFeatures() {
            return features;
        }

        /**
         * Get the maximum distance from this exemplar to the edge of the cluster.
         * <p>
         * For models trained in 4.2 this will return {@link Double#NEGATIVE_INFINITY} as that information is 
         * not produced by 4.2 models.
         * @return The distance to the edge of the cluster.
         */
        public Double getMaxDistToEdge() {
            if (maxDistToEdge != null) {
                return maxDistToEdge;
            }
            else {
                return Double.NEGATIVE_INFINITY;
            }
        }

        /**
         * Copies this cluster exemplar.
         * @return A deep copy of this cluster exemplar.
         */
        public ClusterExemplar copy() {
            return new ClusterExemplar(label,outlierScore,features.copy(),maxDistToEdge);
        }

        @Override
        public String toString() {
            double dist = maxDistToEdge == null ? Double.NEGATIVE_INFINITY : maxDistToEdge;
            return "ClusterExemplar(label="+label+",outlierScore="+outlierScore+",vector="+features+",maxDistToEdge="+dist+")";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ClusterExemplar that = (ClusterExemplar) o;
            return label.equals(that.label) && outlierScore.equals(that.outlierScore) && features.equals(that.features) && Objects.equals(maxDistToEdge, that.maxDistToEdge);
        }

        @Override
        public int hashCode() {
            return Objects.hash(label, outlierScore, features, maxDistToEdge);
        }

        ClusterExemplarProto serialize() {
            ClusterExemplarProto.Builder builder = ClusterExemplarProto.newBuilder();

            builder.setLabel(label);
            builder.setOutlierScore(outlierScore);
            builder.setFeatures(features.serialize());
            builder.setMaxDistToEdge(maxDistToEdge);

            return builder.build();
        }

        static ClusterExemplar deserialize(ClusterExemplarProto proto) {
            Tensor tensor = Tensor.deserialize(proto.getFeatures());
            if (!(tensor instanceof SGDVector)) {
                throw new IllegalStateException("Invalid protobuf, features must be an SGDVector, found " + tensor.getClass());
            }
            SGDVector vector = (SGDVector) tensor;
            return new ClusterExemplar(proto.getLabel(),proto.getOutlierScore(),vector,proto.getMaxDistToEdge());
        }
    }
    
}

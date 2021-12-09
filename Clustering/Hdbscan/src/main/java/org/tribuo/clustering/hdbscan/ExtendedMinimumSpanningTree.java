/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * An Extended Minimum Spanning Tree graph. Includes the functionality to sort the edge weights in ascending order.
 */
final class ExtendedMinimumSpanningTree {
    private final int numVertices;

    private final EMSTTriple[] emstTriples;

    private final List<Integer>[] edges;

    /**
     * Constructs an ExtendedMinimumSpanningTree, including creating an edge list for each vertex from the
     * vertex arrays. For an index i, firstVertices[i] and secondVertices[i] share an edge with weight
     * edgeWeights[i].
     * @param numVertices The number of vertices in the graph (indexed 0 to numVertices-1)
     * @param firstVertices An array of vertices corresponding to the array of edges
     * @param secondVertices An array of vertices corresponding to the array of edges
     * @param edgeWeights An array of edges corresponding to the arrays of vertices
     */
    ExtendedMinimumSpanningTree(int numVertices, int[] firstVertices, int[] secondVertices, double[] edgeWeights) {
        this.numVertices = numVertices;
        // Only integer arraylists are inserted into this array, and it's not accessible outside.
        @SuppressWarnings("unchecked")
        List<Integer>[] edgeTmp = (List<Integer>[]) new ArrayList[numVertices];
        this.edges = edgeTmp;
        for (int i = 0; i < this.edges.length; i++) {
            this.edges[i] = new ArrayList<>(1 + edgeWeights.length / numVertices);
        }

        emstTriples = new EMSTTriple[edgeWeights.length];
        for (int i = 0; i < edgeWeights.length; i++) {
            int vertexOne = firstVertices[i];
            int vertexTwo = secondVertices[i];
            this.edges[vertexOne].add(vertexTwo);
            if (vertexOne != vertexTwo) {
                this.edges[vertexTwo].add(vertexOne);
            }

            EMSTTriple emstTriple = new EMSTTriple(vertexOne, vertexTwo, edgeWeights[i]);
            emstTriples[i] = (emstTriple);
        }

        Arrays.sort(emstTriples);
    }

    public int getNumVertices() {
        return this.numVertices;
    }

    public int getNumEdges() {
        return this.emstTriples.length;
    }

    public int getFirstVertexAtIndex(int index) {
        return this.emstTriples[index].firstVertex;
    }

    public int getSecondVertexAtIndex(int index) {
        return this.emstTriples[index].secondVertex;
    }

    public double getEdgeWeightAtIndex(int index) {
        return this.emstTriples[index].edgeWeight;
    }

    public List<Integer> getEdgeListForVertex(int vertex) {
        return this.edges[vertex];
    }

    /**
     * Encapsulate the vertices and the edge weight together.
     */
    final private static class EMSTTriple implements Comparable<EMSTTriple> {
        final int firstVertex;
        final int secondVertex;
        final double edgeWeight;

        EMSTTriple(int firstVertex, int secondVertex, double edgeWeight) {
            this.firstVertex = firstVertex;
            this.secondVertex = secondVertex;
            this.edgeWeight = edgeWeight;
        }

        @Override
        public int compareTo(EMSTTriple that) {
            return Double.compare(this.edgeWeight, that.edgeWeight);
        }
    }
}
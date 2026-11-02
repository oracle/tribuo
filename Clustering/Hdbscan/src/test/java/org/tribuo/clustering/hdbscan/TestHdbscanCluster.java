/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Test;
import org.tribuo.MutableDataset;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.ClusteringFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.*;

/** Regression coverage for the immutable sentinel and root propagation boundary. */
public class TestHdbscanCluster {
    private static final HdbscanCluster SENTINEL = HdbscanCluster.NOT_A_CLUSTER;

    private static void assertSentinel() {
        assertEquals(-1, SENTINEL.getLabel());
        assertNull(SENTINEL.getParent());
        assertFalse(SENTINEL.hasChildren());
        assertEquals(0, SENTINEL.getHierarchyLevel());
        assertEquals(Double.MAX_VALUE, SENTINEL.getPropagatedLowestChildSplitLevel());
        assertTrue(SENTINEL.getPropagatedDescendants().isEmpty());
    }

    @Test
    public void rejectsMutationsBeforeSideEffects() {
        int[] labels = {7};
        TreeSet<Integer> points = new TreeSet<>(Collections.singleton(0));
        assertThrows(UnsupportedOperationException.class, () -> SENTINEL.createNewCluster(points, labels, 2, 1));
        assertArrayEquals(new int[]{7}, labels);
        assertEquals(Collections.singleton(0), points);
        assertThrows(UnsupportedOperationException.class, () -> SENTINEL.detachPoints(1, 1));
        assertThrows(UnsupportedOperationException.class, SENTINEL::finishPropagation);
        assertThrows(UnsupportedOperationException.class, SENTINEL::propagateToParent);
        assertThrows(UnsupportedOperationException.class, SENTINEL::markHasChildren);
        assertThrows(UnsupportedOperationException.class, () -> SENTINEL.acceptPropagation(1, 1, Collections.singletonList(SENTINEL)));
        assertThrows(UnsupportedOperationException.class, () -> SENTINEL.setHierarchyLevel(1));
        assertThrows(UnsupportedOperationException.class, () -> SENTINEL.getPropagatedDescendants().add(SENTINEL));
        assertSentinel();
    }

    @Test
    public void unsplitRootFinalization() {
        HdbscanCluster root = HdbscanCluster.create(1, SENTINEL, Double.NaN, 3);
        root.detachPoints(3, 2);
        root.finishPropagation();
        assertEquals(2, root.getPropagatedLowestChildSplitLevel());
        assertTrue(root.getPropagatedDescendants().isEmpty());
        assertSentinel();
    }

    @Test
    public void selectsClustersAndDescendantsIncludingTies() {
        for (int stability = 1; stability <= 3; stability++) {
            HdbscanCluster root = HdbscanCluster.create(1, SENTINEL, Double.NaN, 2);
            HdbscanCluster middle = HdbscanCluster.create(2, root, 2, 2);
            HdbscanCluster leaf = HdbscanCluster.create(3, middle, 1, 2);
            leaf.detachPoints(2, 0.5); // Stability 2.
            middle.detachPoints(2, 2.0 / (stability + 1));
            leaf.finishPropagation();
            leaf.propagateToParent();
            middle.finishPropagation();
            middle.propagateToParent();
            root.finishPropagation();
            assertEquals(Collections.singletonList(stability < 2 ? leaf : middle), root.getPropagatedDescendants());
            assertEquals(0.5, root.getPropagatedLowestChildSplitLevel());
            assertTrue(root.hasChildren());
            assertTrue(root.compareTo(middle) > 0);
            assertSentinel();
        }
    }

    private static MutableDataset<ClusterID> dataset() {
        ClusteringFactory factory = new ClusteringFactory();
        return new MutableDataset<>(new SimpleDataSourceProvenance("sentinel-regression", factory), factory);
    }

    private static HdbscanTrainer trainer(int size, int k) {
        return new HdbscanTrainer(size, DistanceType.L2.getDistance(), k, 1, NeighboursQueryFactoryType.BRUTE_FORCE);
    }

    @Test
    public void repeatedTrainingDoesNotRetainClusters() {
        MutableDataset<ClusterID> data = dataset();
        Random random = new Random(42);
        for (int blob = 0; blob < 3; blob++) {
            for (int i = 0; i < 30; i++) {
                data.add(new ArrayExample<>(ClusteringFactory.UNASSIGNED_CLUSTER_ID, new String[]{"x", "y"},
                        new double[]{blob * 100 + random.nextGaussian(), blob * 100 + random.nextGaussian()}));
            }
        }
        HdbscanTrainer reused = trainer(5, 5);
        HdbscanModel baseline = reused.train(data);
        assertSentinel();
        for (int i = 0; i < 5; i++) {
            for (HdbscanTrainer trainer : Arrays.asList(reused, trainer(5, 5))) {
                HdbscanModel model = trainer.train(data);
                assertEquals(baseline.getClusterLabels(), model.getClusterLabels());
                assertEquals(baseline.getOutlierScores(), model.getOutlierScores());
                assertSentinel();
            }
        }
    }

    @Test
    public void unsplitTrainingPreservesOutlierScores() {
        MutableDataset<ClusterID> data = dataset();
        for (double x : new double[]{1, 2, 4}) {
            data.add(new ArrayExample<>(ClusteringFactory.UNASSIGNED_CLUSTER_ID, new String[]{"x"}, new double[]{x}));
        }
        HdbscanModel model = trainer(5, 2).train(data);
        assertEquals(Arrays.asList(0, 0, 0), model.getClusterLabels());
        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, model.getOutlierScores().get(i));
        }
        assertSentinel();
    }
}

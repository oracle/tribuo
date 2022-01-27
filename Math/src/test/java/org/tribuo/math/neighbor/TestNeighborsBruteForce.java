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

package org.tribuo.math.neighbor;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for the brute-force nearest neighbor query implementation.
 */
public class TestNeighborsBruteForce {

    private static final int NUM_NEIGHBORS_K = 3;

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(NeighborsBruteForce.class.getName());
        logger.setLevel(Level.WARNING);
    }

    private static void assertNeighborPoints(SGDVector[] data, List<Pair<Integer, Double>> indexDistancePairList,
                                             double[] point0, double[] point1, double[] point2)  {
        // These tests use NUM_NEIGHBORS_K = 3 throughout, so we check each of the identified neighboring points
        Pair<Integer, Double> indexDistancePair = indexDistancePairList.get(0);
        assertArrayEquals(point0, data[indexDistancePair.getA()].toArray());
        indexDistancePair = indexDistancePairList.get(1);
        assertArrayEquals(point1, data[indexDistancePair.getA()].toArray());
        indexDistancePair = indexDistancePairList.get(2);
        assertArrayEquals(point2, data[indexDistancePair.getA()].toArray());
    }

    private static void assertNeighborDistances(List<Pair<Integer, Double>> indexDistancePairList, double expectedDistance0,
                                                double expectedDistance1, double expectedDistance2)  {

        // These tests use NUM_NEIGHBORS_K = 3 throughout, so we check each of the identified neighboring distances
        Pair<Integer, Double> indexDistancePair = indexDistancePairList.get(0);
        assertEquals(expectedDistance0, indexDistancePair.getB().doubleValue());
        indexDistancePair = indexDistancePairList.get(1);
        assertEquals(expectedDistance1, indexDistancePair.getB().doubleValue());
        indexDistancePair = indexDistancePairList.get(2);
        assertEquals(expectedDistance2, indexDistancePair.getB().doubleValue());
    }

    private static void neighborsBrutForceQueryAll(NeighborsBruteForce nbf, SGDVector[] data) {
        List<Pair<Integer, Double>>[] indexDistancePairListArray = nbf.queryAll(NUM_NEIGHBORS_K);

        //////////////////////////////////////////////
        // Check the first point in the data
        List<Pair<Integer, Double>> indexDistancePairList = indexDistancePairListArray[0];

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] firstExpectedPoint0 = {0.76, 1.0};
        double[] firstExpectedPoint1 = {1.0, 0.75};
        double[] firstExpectedPoint2 = {1.05, 1.0};

        assertNeighborPoints(data, indexDistancePairList, firstExpectedPoint0, firstExpectedPoint1, firstExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        double expectedDistance0 = 0.23021728866442676;
        double expectedDistance1 = 0.25079872407968906;
        double expectedDistance2 = 0.378021163428716;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);

        //////////////////////////////////////////////
        // Check another point in the data
        indexDistancePairList = indexDistancePairListArray[19];

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] secondExpectedPoint0 = {2.7,2.7};
        double[] secondExpectedPoint1 = {2.55,2.6};
        double[] secondExpectedPoint2 = {3.15,2.5};

        assertNeighborPoints(data, indexDistancePairList, secondExpectedPoint0, secondExpectedPoint1, secondExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        expectedDistance0 = 0.22360679774997896;
        expectedDistance1 = 0.26925824035672524;
        expectedDistance2 = 0.3500000000000001;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);

        //////////////////////////////////////////////
        // Check another point in the data
        indexDistancePairList = indexDistancePairListArray[29];

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] thirdExpectedPoint0 = {3.25,3.1};
        double[] thirdExpectedPoint1 = {3.2,3.25};
        double[] thirdExpectedPoint2 = {5.05,3.12};

        assertNeighborPoints(data, indexDistancePairList, thirdExpectedPoint0, thirdExpectedPoint1, thirdExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        expectedDistance0 = 0.8805679985100522;
        expectedDistance1 = 0.9769339793455846;
        expectedDistance2 = 0.9823441352194252;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);

        //////////////////////////////////////////////
        // Check the last point in the data
        indexDistancePairList = indexDistancePairListArray[38];

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] fourthExpectedPoint0 = {5.23,5.02};
        double[] fourthExpectedPoint1 = {4.95,5.25};
        double[] fourthdExpectedPoint2 = {5.01,5.03};

        assertNeighborPoints(data, indexDistancePairList, fourthExpectedPoint0, fourthExpectedPoint1, fourthdExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        expectedDistance0 = 0.2607680962081067;
        expectedDistance1 = 0.26172504656604784;
        expectedDistance2 = 0.32015621187164256;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);
    }

    @Test
    public void testSingleThreadQueryAll() {
        SGDVector[] data = getTestDataVectorArray();
        NeighborsBruteForce nbf = new NeighborsBruteForce(data, DistanceType.EUCLIDEAN, 1);
        neighborsBrutForceQueryAll(nbf, data);
    }

    @Test
    public void testMultiThreadQueryAll() {
        SGDVector[] data = getTestDataVectorArray();
        NeighborsBruteForce nbf = new NeighborsBruteForce(data, DistanceType.EUCLIDEAN, 4);
        neighborsBrutForceQueryAll(nbf, data);
    }

    private static void neighborsBrutForceQueryMany(NeighborsBruteForce nbf, SGDVector[] data, SGDVector[] queryData) {
        List<Pair<Integer, Double>>[] indexDistancePairListArray = nbf.query(queryData, NUM_NEIGHBORS_K);

        //////////////////////////////////////////////
        //Check the first point provided to the query
        List<Pair<Integer, Double>> indexDistancePairList = indexDistancePairListArray[0];

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] firstExpectedPoint0 = {0.76, 1.0};
        double[] firstExpectedPoint1 = {1.0, 0.75};
        double[] firstExpectedPoint2 = {1.05, 1.0};

        assertNeighborPoints(data, indexDistancePairList, firstExpectedPoint0, firstExpectedPoint1, firstExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        double expectedDistance0 = 0.23021728866442676;
        double expectedDistance1 = 0.25079872407968906;
        double expectedDistance2 = 0.378021163428716;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);

        //////////////////////////////////////////////
        // Check the second point provided to the query
        indexDistancePairList = indexDistancePairListArray[1];

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] secondExpectedPoint0 = {2.7,2.7};
        double[] secondExpectedPoint1 = {2.55,2.6};
        double[] secondExpectedPoint2 = {3.15,2.5};

        assertNeighborPoints(data, indexDistancePairList, secondExpectedPoint0, secondExpectedPoint1, secondExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        expectedDistance0 = 0.22360679774997896;
        expectedDistance1 = 0.26925824035672524;
        expectedDistance2 = 0.3500000000000001;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);

        //////////////////////////////////////////////
        // Check the third point provided to the query
        indexDistancePairList = indexDistancePairListArray[2];

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] thirdExpectedPoint0 = {3.25,3.1};
        double[] thirdExpectedPoint1 = {3.2,3.25};
        double[] thirdExpectedPoint2 = {5.05,3.12};

        assertNeighborPoints(data, indexDistancePairList, thirdExpectedPoint0, thirdExpectedPoint1, thirdExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        expectedDistance0 = 0.8805679985100522;
        expectedDistance1 = 0.9769339793455846;
        expectedDistance2 = 0.9823441352194252;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);
    }

    @Test
    public void testSingleThreadQueryMany() {
        SGDVector[] data = getTestDataVectorArray();
        SGDVector[] queryData = getTestQueryVectorArray();
        NeighborsBruteForce nbf = new NeighborsBruteForce(data, DistanceType.EUCLIDEAN, 1);
        neighborsBrutForceQueryMany(nbf, data, queryData);
    }

    @Test
    public void testMultiThreadQueryMany() {
        SGDVector[] data = getTestDataVectorArray();
        SGDVector[] queryData = getTestQueryVectorArray();
        NeighborsBruteForce nbf = new NeighborsBruteForce(data, DistanceType.EUCLIDEAN, 2);
        neighborsBrutForceQueryMany(nbf, data, queryData);
    }

    @Test
    public void testNeighborsBrutForceQueryOne() {
        SGDVector[] data = getTestDataVectorArray();
        SGDVector vector = get2DPoint(5.21,5.28);
        NeighborsBruteForce nbf = new NeighborsBruteForce(data);
        List<Pair<Integer, Double>> indexDistancePairList = nbf.query(vector, NUM_NEIGHBORS_K);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring points
        double[] firstExpectedPoint0 = {5.23,5.02};
        double[] firstExpectedPoint1 = {4.95,5.25};
        double[] firstExpectedPoint2 = {5.01,5.03};

        assertNeighborPoints(data, indexDistancePairList, firstExpectedPoint0, firstExpectedPoint1, firstExpectedPoint2);

        // These tests use NUM_NEIGHBORS_K = 3 throughout. These are the expected neighboring distances
        double expectedDistance0 = 0.2607680962081067;
        double expectedDistance1 = 0.26172504656604784;
        double expectedDistance2 = 0.32015621187164256;

        assertNeighborDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);
    }

    private SGDVector[] getTestDataVectorArray() {
        List<SGDVector> vectorList = new ArrayList<>();

        vectorList.add(get2DPoint(0.75,0.77));
        vectorList.add(get2DPoint(0.76,1));
        vectorList.add(get2DPoint(0.85,1.45));
        vectorList.add(get2DPoint(1,0.75));
        vectorList.add(get2DPoint(1.05,1));
        vectorList.add(get2DPoint(1.07,1.25));
        vectorList.add(get2DPoint(1.2,2.05));
        vectorList.add(get2DPoint(1.25,0.95));
        vectorList.add(get2DPoint(1.27,1.25));
        vectorList.add(get2DPoint(1.3,4.31));
        vectorList.add(get2DPoint(1.4,1.11));
        vectorList.add(get2DPoint(2.2,1.23));
        vectorList.add(get2DPoint(2.27,2.85));
        vectorList.add(get2DPoint(2.4,4.6));
        vectorList.add(get2DPoint(2.5,3.1));
        vectorList.add(get2DPoint(2.55,2.6));
        vectorList.add(get2DPoint(2.6,2.9));
        vectorList.add(get2DPoint(2.7,2.7));
        vectorList.add(get2DPoint(2.78,3));
        vectorList.add(get2DPoint(2.8,2.5));
        vectorList.add(get2DPoint(3,3));
        vectorList.add(get2DPoint(3.05,3.08));
        vectorList.add(get2DPoint(3.05,3.2));
        vectorList.add(get2DPoint(3.1,2.85));
        vectorList.add(get2DPoint(3.15,2.5));
        vectorList.add(get2DPoint(3.2,3.25));
        vectorList.add(get2DPoint(3.25,3.1));
        vectorList.add(get2DPoint(3.65,5.05));
        vectorList.add(get2DPoint(3.8,0.8));
        vectorList.add(get2DPoint(4.1,2.87));
        vectorList.add(get2DPoint(4.77,4.75));
        vectorList.add(get2DPoint(4.74,5.05));
        vectorList.add(get2DPoint(4.8,1.9));
        vectorList.add(get2DPoint(4.95,4.73));
        vectorList.add(get2DPoint(4.95,5.25));
        vectorList.add(get2DPoint(5.01,5.03));
        vectorList.add(get2DPoint(5.05,3.12));
        vectorList.add(get2DPoint(5.23,5.02));
        vectorList.add(get2DPoint(5.21,5.28));

        SGDVector[] vectorArr = new SGDVector[vectorList.size()];
        vectorList.toArray(vectorArr);
        return vectorArr;
    }

    private SGDVector[] getTestQueryVectorArray() {
        List<SGDVector> vectorList = new ArrayList<>();

        vectorList.add(get2DPoint(0.75,0.77));
        vectorList.add(get2DPoint(2.8,2.5));
        vectorList.add(get2DPoint(4.1,2.87));

        SGDVector[] vectorArr = new SGDVector[vectorList.size()];
        vectorList.toArray(vectorArr);
        return vectorArr;
    }

    private SGDVector get2DPoint(double x, double y) {
        return DenseVector.createDenseVector(new double[]{x, y});
    }

}

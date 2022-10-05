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

package org.tribuo.math.neighbour;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.util.SGDVectorsFromCSV;

import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * This helper class contains specific unit test details common to both {@link TestKDTree} and
 * {@link TestNeighborsBruteForce}. Both of these neighbour query implementations produce identical results.
 */
public class NeighbourQueryTestHelper {

    private static final int NUM_NEIGHBOURS_K = 4;

    static void neighboursQueryAll(NeighboursQueryFactory nqf) {
        // Don't use the shuffled data. These assertions depend on points at specific indicies.
        SGDVector[] data = getTestDataVectorArray();

        NeighboursQuery nq = nqf.createNeighboursQuery(data);

        List<List<Pair<Integer, Double>>> indexDistancePairListOfLists = nq.queryAll(NUM_NEIGHBOURS_K);

        //////////////////////////////////////////////
        // Check the first point in the data
        List<Pair<Integer, Double>> indexDistancePairList = indexDistancePairListOfLists.get(0);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] firstExpectedPoint0 = {0.75,0.77};
        double[] firstExpectedPoint1 = {0.76, 1.0};
        double[] firstExpectedPoint2 = {1.0, 0.75};
        double[] firstExpectedPoint3 = {1.05, 1.0};

        assertNeighbourPoints(data, indexDistancePairList, firstExpectedPoint0,
            firstExpectedPoint1, firstExpectedPoint2, firstExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        double expectedDistance0 = 0.0;
        double expectedDistance1 = 0.23021728866442676;
        double expectedDistance2 = 0.25079872407968906;
        double expectedDistance3 = 0.378021163428716;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);

        //////////////////////////////////////////////
        // Check another point in the data
        indexDistancePairList = indexDistancePairListOfLists.get(19);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] secondExpectedPoint0 = {2.8,2.5};
        double[] secondExpectedPoint1 = {2.7,2.7};
        double[] secondExpectedPoint2 = {2.55,2.6};
        double[] secondExpectedPoint3 = {3.15,2.5};

        assertNeighbourPoints(data, indexDistancePairList, secondExpectedPoint0,
            secondExpectedPoint1, secondExpectedPoint2, secondExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances
        expectedDistance0 = 0.0;
        expectedDistance1 = 0.22360679774997896;
        expectedDistance2 = 0.26925824035672524;
        expectedDistance3 = 0.3500000000000001;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);

        //////////////////////////////////////////////
        // Check another point in the data
        indexDistancePairList = indexDistancePairListOfLists.get(29);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] thirdExpectedPoint0 = {4.1,2.87};
        double[] thirdExpectedPoint1 = {3.25,3.1};
        double[] thirdExpectedPoint2 = {3.2,3.25};
        double[] thirdExpectedPoint3 = {5.05,3.12};

        assertNeighbourPoints(data, indexDistancePairList, thirdExpectedPoint0,
            thirdExpectedPoint1, thirdExpectedPoint2, thirdExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 0.8805679985100522;
        expectedDistance2 = 0.9769339793455846;
        expectedDistance3 = 0.9823441352194252;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);

        //////////////////////////////////////////////
        // Check the last point in the data
        indexDistancePairList = indexDistancePairListOfLists.get(38);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] fourthExpectedPoint0 = {5.21,5.28};
        double[] fourthExpectedPoint1 = {5.23,5.02};
        double[] fourthExpectedPoint2 = {4.95,5.25};
        double[] fourthExpectedPoint3 = {5.01,5.03};

        assertNeighbourPoints(data, indexDistancePairList, fourthExpectedPoint0,
            fourthExpectedPoint1, fourthExpectedPoint2, fourthExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 0.2607680962081067;
        expectedDistance2 = 0.26172504656604784;
        expectedDistance3 = 0.32015621187164256;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);
    }

    static void neighboursQueryAllIntegers4D(NeighboursQueryFactory nqf) {
        SGDVector[] data = get4DIntegerTestDataVectorArray();

        NeighboursQuery nq = nqf.createNeighboursQuery(data);

        List<List<Pair<Integer, Double>>> indexDistancePairListOfLists = nq.queryAll(NUM_NEIGHBOURS_K);

        //////////////////////////////////////////////
        // Check the first point in the data (9,8,9,8)
        List<Pair<Integer, Double>> indexDistancePairList = indexDistancePairListOfLists.get(59);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] firstExpectedPoint0 = {9.0,8.0,9.0,8.0};
        double[] firstExpectedPoint1 = {9.0,8.0,9.0,8.0};
        double[] firstExpectedPoint2 = {9.0,9.0,9.0,8.0};
        double[] firstExpectedPoint3 = {9.0,9.0,9.0,9.0};

        assertNeighbourPoints(data, indexDistancePairList, firstExpectedPoint0,
            firstExpectedPoint1, firstExpectedPoint2, firstExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        double expectedDistance0 = 0.0;
        double expectedDistance1 = 0.0;
        double expectedDistance2 = 1.0;
        double expectedDistance3 = 1.4142135623730951;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);

        //////////////////////////////////////////////
        // Check another point in the data (0,0,2,0)
        indexDistancePairList = indexDistancePairListOfLists.get(77);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] secondExpectedPoint0 = {0.0,0.0,2.0,0.0};
        double[] secondExpectedPoint1 = {0.0,1.0,3.0,1.0};
        double[] secondExpectedPoint2 = {1.0,2.0,2.0,0.0};
        double[] secondExpectedPoint3 = {0.0,1.0,3.0,2.0};

        assertNeighbourPoints(data, indexDistancePairList, secondExpectedPoint0,
            secondExpectedPoint1, secondExpectedPoint2, secondExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances
        expectedDistance0 = 0.0;
        expectedDistance1 = 1.7320508075688772;
        expectedDistance2 = 2.23606797749979;
        expectedDistance3 = 2.449489742783178;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);

        //////////////////////////////////////////////
        // Check another point in the data (4,4,4,4)
        indexDistancePairList = indexDistancePairListOfLists.get(707);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] thirdExpectedPoint0 = {4.0,4.0,4.0,4.0};
        double[] thirdExpectedPoint1 = {4.0,4.0,4.0,4.0};
        double[] thirdExpectedPoint2 = {4.0,4.0,4.0,4.0};
        double[] thirdExpectedPoint3 = {4.0,4.0,4.0,3.0};

        assertNeighbourPoints(data, indexDistancePairList, thirdExpectedPoint0,
            thirdExpectedPoint1, thirdExpectedPoint2, thirdExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 0.0;
        expectedDistance2 = 0.0;
        expectedDistance3 = 1.0;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);

        //////////////////////////////////////////////
        // Check the last point in the data (7,7,7,8)
        indexDistancePairList = indexDistancePairListOfLists.get(823);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] fourthExpectedPoint0 = {7.0,7.0,7.0,8.0};
        double[] fourthExpectedPoint1 = {7.0,7.0,6.0,8.0};
        double[] fourthExpectedPoint2 = {7.0,8.0,7.0,9.0};
        double[] fourthExpectedPoint3 = {6.0,8.0,7.0,7.0};

        assertNeighbourPoints(data, indexDistancePairList, fourthExpectedPoint0,
            fourthExpectedPoint1, fourthExpectedPoint2, fourthExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 1.0;
        expectedDistance2 = 1.4142135623730951;
        expectedDistance3 = 1.7320508075688772;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1,
            expectedDistance2, expectedDistance3);
    }

    static void neighboursQueryMany(NeighboursQueryFactory nqf) {
        // The tests should still work regardless of the order of the data - shuffle the array.
        SGDVector[] data = getShuffledTestDataVectorArray();
        SGDVector[] queryData = getTestQueryVectorArray();

        NeighboursQuery nq = nqf.createNeighboursQuery(data);
        List<List<Pair<Integer, Double>>> indexDistancePairListofLists = nq.query(queryData, NUM_NEIGHBOURS_K);

        //////////////////////////////////////////////
        //Check the first point provided to the query
        List<Pair<Integer, Double>> indexDistancePairList = indexDistancePairListofLists.get(0);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] firstExpectedPoint0 = {0.75,0.77};
        double[] firstExpectedPoint1 = {0.76, 1.0};
        double[] firstExpectedPoint2 = {1.0, 0.75};
        double[] firstExpectedPoint3 = {1.05, 1.0};

        assertNeighbourPoints(data, indexDistancePairList, firstExpectedPoint0, firstExpectedPoint1,
            firstExpectedPoint2, firstExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        double expectedDistance0 = 0.0;
        double expectedDistance1 = 0.23021728866442676;
        double expectedDistance2 = 0.25079872407968906;
        double expectedDistance3 = 0.378021163428716;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2,
            expectedDistance3);

        //////////////////////////////////////////////
        // Check the second point provided to the query
        indexDistancePairList = indexDistancePairListofLists.get(1);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] secondExpectedPoint0 = {2.8,2.5};
        double[] secondExpectedPoint1 = {2.7,2.7};
        double[] secondExpectedPoint2 = {2.55,2.6};
        double[] secondExpectedPoint3 = {3.15,2.5};

        assertNeighbourPoints(data, indexDistancePairList, secondExpectedPoint0, secondExpectedPoint1,
            secondExpectedPoint2, secondExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 0.22360679774997896;
        expectedDistance2 = 0.26925824035672524;
        expectedDistance3 = 0.3500000000000001;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2,
            expectedDistance3);

        //////////////////////////////////////////////
        // Check the third point provided to the query
        indexDistancePairList = indexDistancePairListofLists.get(2);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] thirdExpectedPoint0 = {4.1,2.87};
        double[] thirdExpectedPoint1 = {3.25,3.1};
        double[] thirdExpectedPoint2 = {3.2,3.25};
        double[] thirdExpectedPoint3 = {5.05,3.12};

        assertNeighbourPoints(data, indexDistancePairList, thirdExpectedPoint0, thirdExpectedPoint1,
            thirdExpectedPoint2, thirdExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 0.8805679985100522;
        expectedDistance2 = 0.9769339793455846;
        expectedDistance3 = 0.9823441352194252;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2,
            expectedDistance3);
    }

    static void neighboursQueryMany3D(NeighboursQueryFactory nqf) {
        // Tests with a 3 dimensional dataset
        SGDVector[] data = get3DTestDataVectorArray();
        SGDVector[] queryData = get3DTestQueryVectorArray();

        NeighboursQuery nq = nqf.createNeighboursQuery(data);
        List<List<Pair<Integer, Double>>> indexDistancePairListofLists = nq.query(queryData, NUM_NEIGHBOURS_K);

        //////////////////////////////////////////////
        //Check the first point provided to the query
        List<Pair<Integer, Double>> indexDistancePairList = indexDistancePairListofLists.get(0);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] firstExpectedPoint0 = {0.9054637330934994,8.682868371123128,9.074535363711819};
        double[] firstExpectedPoint1 = {0.6977937277997284,8.811841760161542,9.450413547361899};
        double[] firstExpectedPoint2 = {0.8618977494727522,8.594289564664695,8.563324968951003};
        double[] firstExpectedPoint3 = {0.3622032035520937,8.399670656036411,9.170818175807304};

        assertNeighbourPoints(data, indexDistancePairList, firstExpectedPoint0, firstExpectedPoint1,
            firstExpectedPoint2, firstExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        double expectedDistance0 = 0.0;
        double expectedDistance1 = 0.4483808371494621;
        double expectedDistance2 = 0.5206536925768924;
        double expectedDistance3 = 0.6201639530744752;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2,
            expectedDistance3);

        //////////////////////////////////////////////
        // Check the second point provided to the query
        indexDistancePairList = indexDistancePairListofLists.get(1);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] secondExpectedPoint0 = {0.808055855779421,4.681697429186865,3.379626271428717};
        double[] secondExpectedPoint1 = {0.7166930967124556,4.668268576688895,3.5265894775752162};
        double[] secondExpectedPoint2 = {0.7984569348434825,4.554785443456714,3.110025446613161};
        double[] secondExpectedPoint3 = {0.44466417053504015,4.669636115365249,3.3530927884064634};

        assertNeighbourPoints(data, indexDistancePairList, secondExpectedPoint0, secondExpectedPoint1,
            secondExpectedPoint2, secondExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 0.1735674848138299;
        expectedDistance2 = 0.2981331852483595;
        expectedDistance3 = 0.3645586618325601;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2,
            expectedDistance3);

        //////////////////////////////////////////////
        // Check the third point provided to the query
        indexDistancePairList = indexDistancePairListofLists.get(2);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring points.
        double[] thirdExpectedPoint0 = {-4.9465405492445775,5.106369933899971,1.3203968622480786};
        double[] thirdExpectedPoint1 = {-3.889807252564182,5.3233994126494535,1.2616644314253151};
        double[] thirdExpectedPoint2 = {-4.087190588116999,4.37117531699009,1.1102140958289508};
        double[] thirdExpectedPoint3 = {-4.272052051675008,5.9344058212416435,1.8597348399857985};

        assertNeighbourPoints(data, indexDistancePairList, thirdExpectedPoint0, thirdExpectedPoint1,
            thirdExpectedPoint2, thirdExpectedPoint3);

        // These tests use NUM_NEIGHBOURS_K = 4 throughout. These are the expected neighboring distances.
        expectedDistance0 = 0.0;
        expectedDistance1 = 1.08038722381822;
        expectedDistance2 = 1.150291387311421;
        expectedDistance3 = 1.1964378873596324;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2,
            expectedDistance3);
    }

    static void neighboursQueryOneInclusive(NeighboursQueryFactory nqf) {
        SGDVector[] data = getTestDataVectorArray();
        // This point is included in the set of points being queried
        SGDVector vector = get2DPoint(5.21,5.28);

        NeighboursQuery nq = nqf.createNeighboursQuery(data);
        List<Pair<Integer, Double>> indexDistancePairList = nq.query(vector, 3);

        // This helper uses k = 3. These are the expected neighboring points.
        // The point itself is returned as the first neighbouring point, so we don't check that.
        double[] firstExpectedPoint0 = {5.21,5.28};
        double[] firstExpectedPoint1 = {5.23,5.02};
        double[] firstExpectedPoint2 = {4.95,5.25};


        assertNeighbourPoints(data, indexDistancePairList, firstExpectedPoint0, firstExpectedPoint1, firstExpectedPoint2);

        // This helper uses k = 3. These are the expected neighboring distances.
        double expectedDistance0 = 0.0;
        double expectedDistance1 = 0.2607680962081067;
        double expectedDistance2 = 0.26172504656604784;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);
    }

    static void neighboursQueryOneExclusive(NeighboursQueryFactory nqf) {
        SGDVector[] data = getShuffledTestDataVectorArray();
        // This point is excluded from the set of points being queried
        SGDVector vector = get2DPoint(5.22,5.25);

        NeighboursQuery nq = nqf.createNeighboursQuery(data);
        List<Pair<Integer, Double>> indexDistancePairList = nq.query(vector, 3);

        // This helper uses k = 3. These are the expected neighboring points.
        // The point itself is returned as the first neighbouring point, so we don't check that.
        double[] firstExpectedPoint0 = {5.21,5.28};
        double[] firstExpectedPoint1 = {5.23,5.02};
        double[] firstExpectedPoint2 = {4.95,5.25};


        assertNeighbourPoints(data, indexDistancePairList, firstExpectedPoint0, firstExpectedPoint1, firstExpectedPoint2);

        // This helper uses k = 3. These are the expected neighboring distances.
        double expectedDistance0 = 0.031622776601683965;
        double expectedDistance1 = 0.23021728866442723;
        double expectedDistance2 = 0.2699999999999996;

        assertNeighbourDistances(indexDistancePairList, expectedDistance0, expectedDistance1, expectedDistance2);
    }

    static void neighboursQuerySingleDimension(NeighboursQueryFactory nqf) {
        SGDVector[] data = new SGDVector[10];
        data[0] = DenseVector.createDenseVector(new double[]{0});
        data[1] = DenseVector.createDenseVector(new double[]{1});
        data[2] = DenseVector.createDenseVector(new double[]{2});
        data[3] = DenseVector.createDenseVector(new double[]{3});
        data[4] = DenseVector.createDenseVector(new double[]{4});
        data[5] = DenseVector.createDenseVector(new double[]{5});
        data[6] = DenseVector.createDenseVector(new double[]{6});
        data[7] = DenseVector.createDenseVector(new double[]{7});
        data[8] = DenseVector.createDenseVector(new double[]{8});
        data[9] = DenseVector.createDenseVector(new double[]{9});

        NeighboursQuery nq = nqf.createNeighboursQuery(data);

        SGDVector candidate = DenseVector.createDenseVector(new double[]{1.75});

        List<Pair<Integer,Double>> query = nq.query(candidate,3);

        assertEquals(2,query.get(0).getA());
        assertEquals(1,query.get(1).getA());
        assertEquals(3,query.get(2).getA());
    }

    private static void assertNeighbourPoints(SGDVector[] data, List<Pair<Integer, Double>> indexDistancePairList, double[]... points)  {
        int i = 0;
        for (double[] point : points) {
            Pair<Integer, Double> indexDistancePair = indexDistancePairList.get(i);
            assertArrayEquals(point, data[indexDistancePair.getA()].toArray());
            i++;
        }
    }

    private static void assertNeighbourDistances(List<Pair<Integer, Double>> indexDistancePairList, double... distances)  {

        int i = 0;
        for (double distance : distances) {
            Pair<Integer, Double> indexDistancePair = indexDistancePairList.get(i);
            assertEquals(distance, indexDistancePair.getB().doubleValue());
            i++;
        }
    }

    private static SGDVector[] get3DTestDataVectorArray() {
        String filename = "basic-gaussians-3d.csv";
        URL filepath = NeighbourQueryTestHelper.class.getClassLoader().getResource(filename);
        return SGDVectorsFromCSV.getSGDVectorsFromCSV(filepath, true);
    }

    private static SGDVector[] get4DIntegerTestDataVectorArray() {
        String filename = "integers-1K-4features.csv";
        URL filepath = NeighbourQueryTestHelper.class.getClassLoader().getResource(filename);
        return SGDVectorsFromCSV.getSGDVectorsFromCSV(filepath, true);
    }

    private static SGDVector[] getTestDataVectorArray() {
        return getTestDataVectorList().toArray(new SGDVector[0]);
    }

    private static SGDVector[] getShuffledTestDataVectorArray() {
        List<SGDVector> vectorList = getTestDataVectorList();
        Collections.shuffle(vectorList);
        return vectorList.toArray(new SGDVector[0]);
    }

    private static List<SGDVector> getTestDataVectorList() {
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

        return vectorList;
    }

    private static SGDVector[] get3DTestQueryVectorArray() {
        List<SGDVector> vectorList = new ArrayList<>();

        vectorList.add(get3DPoint(0.9054637330934994,8.682868371123128,9.074535363711819));
        vectorList.add(get3DPoint(0.808055855779421,4.681697429186865,3.379626271428717));
        vectorList.add(get3DPoint(-4.9465405492445775,5.106369933899971,1.3203968622480786));

        return vectorList.toArray(new SGDVector[0]);
    }

    private static SGDVector get3DPoint(double x, double y, double z) {
        return DenseVector.createDenseVector(new double[]{x, y, z});
    }

    private static SGDVector[] getTestQueryVectorArray() {
        List<SGDVector> vectorList = new ArrayList<>();

        vectorList.add(get2DPoint(0.75,0.77));
        vectorList.add(get2DPoint(2.8,2.5));
        vectorList.add(get2DPoint(4.1,2.87));

        return vectorList.toArray(new SGDVector[0]);
    }

    private static SGDVector get2DPoint(double x, double y) {
        return DenseVector.createDenseVector(new double[]{x, y});
    }
}

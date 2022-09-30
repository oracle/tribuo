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

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForce;
import org.tribuo.math.neighbour.bruteforce.NeighboursBruteForceFactory;
import org.tribuo.math.protos.NeighbourFactoryProto;
import org.tribuo.protos.ProtoUtil;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Unit tests for the brute-force nearest neighbour query implementation.
 */
public class TestNeighborsBruteForce {

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(NeighboursBruteForce.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void testSingleThreadQueryAll() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 1);
        NeighbourQueryTestHelper.neighboursQueryAll(factory);
    }

    @Test
    public void testMultiThreadQueryAll() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 4);
        NeighbourQueryTestHelper.neighboursQueryAll(factory);
    }


    @Test
    public void testSingleThreadQueryMany() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 1);
        NeighbourQueryTestHelper.neighboursQueryMany(factory);
    }

    @Test
    public void testMultiThreadQueryMany() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 2);
        NeighbourQueryTestHelper.neighboursQueryMany(factory);
    }

    @Test
    public void testNeighboursQueryOneInclusive() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 1);
        NeighbourQueryTestHelper.neighboursQueryOneInclusive(factory);
    }

    @Test
    public void testNeighboursQueryOneExclusive() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 1);
        NeighbourQueryTestHelper.neighboursQueryOneExclusive(factory);
    }

    @Test
    public void testSingleDimension() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 1);
        NeighbourQueryTestHelper.neighboursQuerySingleDimension(factory);
    }

    @Test
    public void testMultiThreadQueryMany3D() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 2);
        NeighbourQueryTestHelper.neighboursQueryMany3D(factory);
    }

    @Test
    public void testMultiThreadQueryAllIntegers4D() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 4);
        NeighbourQueryTestHelper.neighboursQueryAllIntegers4D(factory);
    }

    @Test
    public void testInvalidKDTreeFactory() {
        assertThrows(PropertyException.class, () -> new NeighboursBruteForceFactory(DistanceType.L1.getDistance(), 0) );
    }

    @Test
    public void serializationTest() {
        NeighboursBruteForceFactory factory = new NeighboursBruteForceFactory(DistanceType.L2.getDistance(), 4);
        NeighbourFactoryProto proto = factory.serialize();
        NeighboursQueryFactory deser = ProtoUtil.deserialize(proto);
        assertEquals(factory,deser);
    }
}

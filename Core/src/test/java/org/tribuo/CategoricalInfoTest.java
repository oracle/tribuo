/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo;

import org.junit.jupiter.api.Test;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.CategoricalIDInfoProto;
import org.tribuo.protos.core.CategoricalInfoProto;
import org.tribuo.protos.core.VariableInfoProto;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class CategoricalInfoTest {

    private static final double DELTA = 1e-12;

    private static final int NUM_SAMPLES = 5000;

    public static CategoricalInfo generateFullInfo() {
        CategoricalInfo newInfo = new CategoricalInfo("test");

        for (int i = 0; i < 5; i++) {
            newInfo.observe(-1.0);
            newInfo.observe(2.0);
            newInfo.observe(3.0);
            newInfo.observe(4.0);
        }

        return newInfo;
    }

    /**
     * Generates an emtpy info which hasn't observed any values.
     * @return An empty info
     */
    public static CategoricalInfo generateEmptyInfo() {
        return new CategoricalInfo("empty");
    }

    public static CategoricalInfo generateOneValueInfo() {
        CategoricalInfo newInfo = new CategoricalInfo("one-value");

        for (int i = 0; i < 25; i++) {
            newInfo.observe(5);
        }

        return newInfo;
    }

    public static CategoricalInfo generateProtoTestInfo() {
        CategoricalInfo ci = new CategoricalInfo("cat");

        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                ci.observe(i);
            });
        });

         return ci;
    }

    public void checkValueAndProb(CategoricalInfo c, double value, double probability) {
        int idx = -1;
        for (int i = 0; i < c.values.length; i++) {
            if (Math.abs(c.values[i] - value) < DELTA) {
                if (idx > 0) {
                    fail("Found value " + value + " at " + idx + " and " + i);
                } else {
                    idx = i;
                }
            }
        }
        double testProb = idx == 0 ? c.cdf[0] : c.cdf[idx] - c.cdf[idx-1];
        assertEquals(probability, testProb, DELTA);
    }

    @Test
    public void samplingTest() {
        SplittableRandom rng = new SplittableRandom(1);

        CategoricalInfo c;

        c = generateEmptyInfo();

        c.frequencyBasedSample(rng, 50);

        assertEquals(1,c.values.length);
        assertEquals(0,c.values[0],DELTA);
        assertEquals(1,c.cdf.length);
        assertEquals(1.0,c.cdf[0],DELTA);

        for (int i = 0; i < 50; i++) {
            assertEquals(0.0,c.frequencyBasedSample(rng, 50),DELTA);
        }

        c = generateOneValueInfo();

        c.frequencyBasedSample(rng, 50);

        assertEquals(2,c.values.length);
        assertEquals(0,c.values[0],DELTA);
        assertEquals(5,c.values[1],DELTA);
        assertEquals(2,c.values.length);
        assertEquals(0.5,c.cdf[0],DELTA);
        assertEquals(1.0,c.cdf[1],DELTA);

        double sum = 0;
        double posCount = 0.0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            double sample = c.frequencyBasedSample(rng, 50);
            if (sample > DELTA) {
                posCount++;
            }
            sum += sample;
        }
        assertEquals(0.5,posCount/NUM_SAMPLES,1e-1);
        assertEquals(2.5,sum/NUM_SAMPLES,1e-1);

        c = generateOneValueInfo();

        c.frequencyBasedSample(rng,1000);

        assertEquals(2,c.values.length);
        assertEquals(0,c.values[0],DELTA);
        assertEquals(5,c.values[1],DELTA);
        assertEquals(2,c.values.length);
        assertEquals(0.975,c.cdf[0],DELTA);
        assertEquals(1.0,c.cdf[1],DELTA);

        c = generateFullInfo();

        c.frequencyBasedSample(rng,50);

        assertEquals(5,c.values.length);
        assertEquals(5,c.cdf.length);
        checkValueAndProb(c,0.0,0.6);
        checkValueAndProb(c,-1.0,0.1);
        checkValueAndProb(c,2.0,0.1);
        checkValueAndProb(c,3.0,0.1);
        checkValueAndProb(c,4.0,0.1);

        c = generateFullInfo();

        c.frequencyBasedSample(rng, 100);

        assertEquals(5,c.values.length);
        assertEquals(5,c.cdf.length);
        checkValueAndProb(c,0.0,0.8);
        checkValueAndProb(c,-1.0,0.05);
        checkValueAndProb(c,2.0,0.05);
        checkValueAndProb(c,3.0,0.05);
        checkValueAndProb(c,4.0,0.05);
    }

    @Test
    public void equalityTest() {
        CategoricalInfo fullFirst = generateFullInfo();
        CategoricalInfo fullSecond = generateFullInfo();
        CategoricalInfo emptyFirst = generateEmptyInfo();
        CategoricalInfo emptySecond = generateEmptyInfo();
        CategoricalInfo oneFirst = generateOneValueInfo();
        CategoricalInfo oneSecond = generateOneValueInfo();

        assertEquals(fullFirst,fullSecond);
        assertEquals(emptyFirst,emptySecond);
        assertEquals(oneFirst,oneSecond);

        assertNotEquals(fullFirst,emptyFirst);
        assertNotEquals(fullFirst,oneFirst);
    }

    @Test
    void testCategoricalInfo() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                info.observe(i);
            });
        });

        VariableInfoProto infoProto = info.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalInfo", infoProto.getClassName());
        CategoricalInfoProto proto = infoProto.getSerializedData().unpack(CategoricalInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(90, proto.getCount());
        assertEquals(0, proto.getObservedCount());
        assertEquals(Double.NaN, proto.getObservedValue());

        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(9, keyList.size());
        assertEquals(9, valueList.size());

        Map<Double, Long> expectedCounts = new HashMap<>();
        IntStream.range(0, 10).forEach(i -> {
            long count = info.getObservationCount(i);
            expectedCounts.put((double)i, count);
        });

        for (int i=0; i<keyList.size(); i++) {
            assertEquals(expectedCounts.get(keyList.get(i)), valueList.get(i));
        }

        VariableInfo infoD = ProtoUtil.deserialize(infoProto);
        assertEquals(info, infoD);
    }

    @Test
    void testCategoricalInfo2() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            info.observe(5);
        });

        VariableInfoProto infoProto = info.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalInfo", infoProto.getClassName());
        CategoricalInfoProto proto = infoProto.getSerializedData().unpack(CategoricalInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(10, proto.getCount());

        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(0, keyList.size());
        assertEquals(0, valueList.size());
        assertEquals(5, proto.getObservedValue());
        assertEquals(10, proto.getObservedCount());

        VariableInfo infoD = ProtoUtil.deserialize(infoProto);
        assertEquals(info, infoD);
    }

    @Test
    void testCategoricalIdInfo() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                info.observe(i);
            });
        });

        CategoricalIDInfo idInfo = info.makeIDInfo(12345);

        VariableInfoProto infoProto = idInfo.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalIDInfo", infoProto.getClassName());
        CategoricalIDInfoProto proto = infoProto.getSerializedData().unpack(CategoricalIDInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(90, proto.getCount());
        assertEquals(12345, proto.getId());
        assertEquals(0, proto.getObservedCount());
        assertEquals(Double.NaN, proto.getObservedValue());

        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(keyList.size(), valueList.size());

        Map<Double, Long> expectedCounts = new HashMap<>();
        IntStream.range(0, 10).forEach(i -> {
            long count = idInfo.getObservationCount(i);
            expectedCounts.put((double)i, count);
        });

        for (int i=0; i<keyList.size(); i++) {
            assertEquals(expectedCounts.get(keyList.get(i)), valueList.get(i));
        }

        VariableInfo idInfoD = ProtoUtil.deserialize(infoProto);
        assertEquals(idInfo, idInfoD);
    }
}

/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.protos;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.MutableFeatureMap;
import org.tribuo.ensemble.WeightedEnsembleModel;
import org.tribuo.test.MockDataSource;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputInfo;
import org.tribuo.test.MockTrainer;
import org.tribuo.test.MockVotingCombiner;

import java.util.List;

public class ProtoDeserializationCacheTest {

    @Test
    public void testOutputInfoCanonicalise() {
        MockOutput a = new MockOutput("a");
        MockOutput b = new MockOutput("b");
        MockOutput c = new MockOutput("c");

        MockOutputInfo aInfo = new MockOutputInfo();
        MockOutputInfo bInfo = new MockOutputInfo();
        MockOutputInfo cInfo = new MockOutputInfo();

        // aInfo has 4 a and 2 b
        aInfo.observe(a);
        aInfo.observe(a);
        aInfo.observe(a);
        aInfo.observe(a);
        aInfo.observe(b);
        aInfo.observe(b);

        // bInfo has 2 a and 4 b
        bInfo.observe(a);
        bInfo.observe(a);
        bInfo.observe(b);
        bInfo.observe(b);
        bInfo.observe(b);
        bInfo.observe(b);

        // cInfo has 2 a, 2 b and 1 c
        cInfo.observe(a);
        cInfo.observe(a);
        cInfo.observe(b);
        cInfo.observe(b);
        cInfo.observe(c);

        MockOutputInfo aInfoDup = aInfo.copy();
        Assertions.assertNotSame(aInfo, aInfoDup);
        Assertions.assertEquals(aInfo, aInfoDup);

        ProtoDeserializationCache deserCache = new ProtoDeserializationCache();

        var aCan = deserCache.canonicalise(aInfo);
        var aDupCan = deserCache.canonicalise(aInfoDup);
        var bCan = deserCache.canonicalise(bInfo);
        var cCan = deserCache.canonicalise(cInfo);
        Assertions.assertEquals(aCan, aDupCan);
        Assertions.assertSame(aCan, aDupCan);
        Assertions.assertNotSame(aCan, bCan);
        Assertions.assertNotEquals(aCan, bCan);
        Assertions.assertNotSame(aCan, cCan);
        Assertions.assertNotEquals(aCan, cCan);
        Assertions.assertNotSame(bCan, cCan);
        Assertions.assertNotEquals(bCan, cCan);

        Assertions.assertEquals(3, deserCache.outputInfoCacheSize());
        Assertions.assertEquals(0, deserCache.featureMapCacheSize());
    }

    @Test
    public void testFeatureMapCanonicalise() {
        Feature a5 = new Feature("a", 5.0);
        Feature a2 = new Feature("a", 2.0);
        Feature b = new Feature("b", 3.0);
        Feature c = new Feature("c", 1.0);

        MutableFeatureMap aMap = new MutableFeatureMap();
        MutableFeatureMap bMap = new MutableFeatureMap();
        MutableFeatureMap cMap = new MutableFeatureMap();

        // aInfo has 4 a and 2 b
        aMap.add(a5);
        aMap.add(a5);
        aMap.add(a2);
        aMap.add(a2);
        aMap.add(b);
        aMap.add(b);

        // bInfo has 4 a and 2 b
        bMap.add(a2);
        bMap.add(a2);
        bMap.add(a2);
        bMap.add(a2);
        bMap.add(b);
        bMap.add(b);

        // cInfo has 2 a, 2 b and 1 c
        cMap.add(a5);
        cMap.add(a2);
        cMap.add(b);
        cMap.add(b);
        cMap.add(c);

        ImmutableFeatureMap aImMap = new ImmutableFeatureMap(aMap);
        ImmutableFeatureMap bImMap = new ImmutableFeatureMap(bMap);
        ImmutableFeatureMap cImMap = new ImmutableFeatureMap(cMap);

        ImmutableFeatureMap aImMapDup = new ImmutableFeatureMap(aMap);
        Assertions.assertNotSame(aImMap, aImMapDup);
        Assertions.assertEquals(aImMap, aImMapDup);

        ProtoDeserializationCache deserCache = new ProtoDeserializationCache();

        var aCan = deserCache.canonicalise(aImMap);
        var aDupCan = deserCache.canonicalise(aImMapDup);
        var bCan = deserCache.canonicalise(bImMap);
        var cCan = deserCache.canonicalise(cImMap);
        Assertions.assertEquals(aCan, aDupCan);
        Assertions.assertSame(aCan, aDupCan);
        Assertions.assertNotSame(aCan, bCan);
        Assertions.assertNotEquals(aCan, bCan);
        Assertions.assertNotSame(aCan, cCan);
        Assertions.assertNotEquals(aCan, cCan);
        Assertions.assertNotSame(bCan, cCan);
        Assertions.assertNotEquals(bCan, cCan);

        Assertions.assertEquals(0, deserCache.outputInfoCacheSize());
        Assertions.assertEquals(3, deserCache.featureMapCacheSize());
    }

    @Test
    public void testDeduplication() {
        MockTrainer t = new MockTrainer("A");

        MockDataSource source = new MockDataSource(10);
        MutableDataset<MockOutput> dataset = new MutableDataset<>(source);

        MockDataSource otherSource = new MockDataSource(10);
        MutableDataset<MockOutput> datasetDup = new MutableDataset<>(otherSource);

        Assertions.assertEquals(dataset.getFeatureMap(), datasetDup.getFeatureMap());
        Assertions.assertNotSame(dataset.getFeatureMap(), datasetDup.getFeatureMap());
        Assertions.assertEquals(dataset.getOutputInfo(), datasetDup.getOutputInfo());
        Assertions.assertNotSame(dataset.getOutputInfo(), datasetDup.getOutputInfo());

        var first = t.train(dataset);
        var second = t.train(datasetDup);

        Assertions.assertEquals(first.getFeatureIDMap(), second.getFeatureIDMap());
        Assertions.assertNotSame(first.getFeatureIDMap(), second.getFeatureIDMap());
        Assertions.assertEquals(first.getOutputIDInfo(), second.getOutputIDInfo());
        Assertions.assertNotSame(first.getOutputIDInfo(), second.getOutputIDInfo());

        // WeightedEnsembleModel uses the first model's feature and output info directly, but
        // doesn't dedup it with the other ensemble members as models are immutable.
        var ensemble = WeightedEnsembleModel.createEnsembleFromExistingModels("test-ensemble", List.of(first, second), new MockVotingCombiner());
        Assertions.assertSame(ensemble.getFeatureIDMap(), first.getFeatureIDMap());
        Assertions.assertSame(ensemble.getOutputIDInfo(), first.getOutputIDInfo());
        Assertions.assertNotSame(ensemble.getFeatureIDMap(), second.getFeatureIDMap());
        Assertions.assertNotSame(ensemble.getOutputIDInfo(), second.getOutputIDInfo());

        // Test deduplication of feature and output infos.
        var deserEnsemble = (WeightedEnsembleModel<?>) Model.deserialize(ensemble.serialize());
        var deserModels = deserEnsemble.getModels();
        Assertions.assertSame(deserEnsemble.getFeatureIDMap(), deserModels.get(0).getFeatureIDMap());
        Assertions.assertSame(deserEnsemble.getOutputIDInfo(), deserModels.get(0).getOutputIDInfo());
        Assertions.assertSame(deserEnsemble.getFeatureIDMap(), deserModels.get(1).getFeatureIDMap());
        Assertions.assertSame(deserEnsemble.getOutputIDInfo(), deserModels.get(1).getOutputIDInfo());
        Assertions.assertSame(deserModels.get(0).getFeatureIDMap(), deserModels.get(1).getFeatureIDMap());
        Assertions.assertSame(deserModels.get(0).getOutputIDInfo(), deserModels.get(1).getOutputIDInfo());

        // Check against original
        Assertions.assertEquals(deserEnsemble.getFeatureIDMap(), ensemble.getFeatureIDMap());
        Assertions.assertEquals(deserModels.get(0).getFeatureIDMap(), first.getFeatureIDMap());
        Assertions.assertEquals(deserModels.get(1).getFeatureIDMap(), second.getFeatureIDMap());
        Assertions.assertNotSame(deserEnsemble.getFeatureIDMap(), ensemble.getFeatureIDMap());
        Assertions.assertNotSame(deserModels.get(0).getFeatureIDMap(), first.getFeatureIDMap());
        Assertions.assertNotSame(deserModels.get(1).getFeatureIDMap(), second.getFeatureIDMap());
        Assertions.assertEquals(deserEnsemble.getOutputIDInfo(), ensemble.getOutputIDInfo());
        Assertions.assertEquals(deserModels.get(0).getOutputIDInfo(), first.getOutputIDInfo());
        Assertions.assertEquals(deserModels.get(1).getOutputIDInfo(), second.getOutputIDInfo());
        Assertions.assertNotSame(deserEnsemble.getOutputIDInfo(), ensemble.getOutputIDInfo());
        Assertions.assertNotSame(deserModels.get(0).getOutputIDInfo(), first.getOutputIDInfo());
        Assertions.assertNotSame(deserModels.get(1).getOutputIDInfo(), second.getOutputIDInfo());
    }

}

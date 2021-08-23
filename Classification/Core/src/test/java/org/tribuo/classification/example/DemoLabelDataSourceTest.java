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

package org.tribuo.classification.example;

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.test.Helpers;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class DemoLabelDataSourceTest {

    @Test
    public void testCheckerboard() {
        // Check zero samples throws
        assertThrows(PropertyException.class, () -> new CheckerboardDataSource(0, 1, 10, 0.0, 1.0));
        // Check invalid numSquares throws
        assertThrows(PropertyException.class, () -> new CheckerboardDataSource(200, 1, 1, 0.0, 1.0));
        // Check invalid min & max
        assertThrows(PropertyException.class, () -> new CheckerboardDataSource(200, 1, 10, 0.0, 0.0));
        assertThrows(PropertyException.class, () -> new CheckerboardDataSource(200, 1, 10, 0.0, -1.0));

        // Check valid parameters work
        CheckerboardDataSource source = new CheckerboardDataSource(2000, 1, 10, -1.0, 1.0);
        assertEquals(2000, source.examples.size());
        Dataset<Label> dataset = new MutableDataset<>(source);
        Map<String, Long> map = new HashMap<>();
        dataset.getOutputInfo().outputCountsIterable().forEach((p) -> map.put(p.getA(), p.getB()));
        assertEquals(996, map.get("X"));
        assertEquals(1004, map.get("O"));
        Helpers.testProvenanceMarshalling(source.getProvenance());
    }

    @Test
    public void testConcentricCircles() {
        // Check zero samples throws
        assertThrows(PropertyException.class, () -> new ConcentricCirclesDataSource(0, 1, 1, 0.5));
        // Check invalid radius throws
        assertThrows(PropertyException.class, () -> new ConcentricCirclesDataSource(200, 1, -1, 0.5));
        // Check invalid class proportion throws
        assertThrows(PropertyException.class, () -> new ConcentricCirclesDataSource(200, 1, 1, 0.0));

        // Check valid parameters work
        ConcentricCirclesDataSource source = new ConcentricCirclesDataSource(200, 1, 1, 0.5);
        assertEquals(200, source.examples.size());
        Dataset<Label> dataset = new MutableDataset<>(source);
        Map<String, Long> map = new HashMap<>();
        dataset.getOutputInfo().outputCountsIterable().forEach((p) -> map.put(p.getA(), p.getB()));
        assertEquals(94, map.get("X"));
        assertEquals(106, map.get("O"));
        Helpers.testProvenanceMarshalling(source.getProvenance());
    }

    @Test
    public void testGaussianGenerator() {
        double[] firstMean = new double[]{0.0, 0.0};
        double[] firstCovariance = new double[]{1.0, 0.1, 0.1, 1.0};
        double[] secondMean = new double[]{1.0, 1.0};
        double[] secondCovariance = new double[]{1.0, 0.5, 0.5, 1.0};
        // Check zero samples throws
        assertThrows(PropertyException.class, () -> new GaussianLabelDataSource(0, 1, firstMean, firstCovariance, secondMean, secondCovariance));
        // Check invalid mean throws
        assertThrows(PropertyException.class, () -> new GaussianLabelDataSource(200, 1, new double[]{0.0}, firstCovariance, secondMean, secondCovariance));
        assertThrows(PropertyException.class, () -> new GaussianLabelDataSource(200, 1, firstMean, firstCovariance, new double[]{0.0, 0.0, 0.0}, secondCovariance));
        // Check invalid covariance throws
        assertThrows(PropertyException.class, () -> new GaussianLabelDataSource(200, 1, firstMean, new double[]{0.1}, secondMean, secondCovariance));
        assertThrows(PropertyException.class, () -> new GaussianLabelDataSource(200, 1, firstMean, firstCovariance, secondMean, new double[]{0.1}));
        assertThrows(PropertyException.class, () -> new GaussianLabelDataSource(200, 1, new double[]{0.1, 0.0, 1.0, 0.1}, firstCovariance, secondMean, secondCovariance));
        assertThrows(PropertyException.class, () -> new GaussianLabelDataSource(200, 1, firstMean, firstCovariance, secondMean, new double[]{0.1, 0.0, 0.0, -1.0}));

        // Check valid parameters work
        GaussianLabelDataSource source = new GaussianLabelDataSource(200, 1, firstMean, firstCovariance, secondMean, secondCovariance);
        assertEquals(200, source.examples.size());
        Dataset<Label> dataset = new MutableDataset<>(source);
        Map<String, Long> map = new HashMap<>();
        dataset.getOutputInfo().outputCountsIterable().forEach((p) -> map.put(p.getA(), p.getB()));
        assertEquals(100, map.get("X"));
        assertEquals(100, map.get("O"));
        Helpers.testProvenanceMarshalling(source.getProvenance());
    }

    @Test
    public void testInterlockingCrescents() {
        // Check zero samples throws
        assertThrows(PropertyException.class, () -> new InterlockingCrescentsDataSource(0));

        // Check valid parameters work
        InterlockingCrescentsDataSource source = new InterlockingCrescentsDataSource(200);
        assertEquals(200, source.examples.size());
        Dataset<Label> dataset = new MutableDataset<>(source);
        Map<String, Long> map = new HashMap<>();
        dataset.getOutputInfo().outputCountsIterable().forEach((p) -> map.put(p.getA(), p.getB()));
        assertEquals(100, map.get("X"));
        assertEquals(100, map.get("O"));
        Helpers.testProvenanceMarshalling(source.getProvenance());
    }

    @Test
    public void testNoisyInterlockingCrescents() {
        // Check zero samples throws
        assertThrows(PropertyException.class, () -> new NoisyInterlockingCrescentsDataSource(0, 1, 0.1));
        // Check invalid variance throws
        assertThrows(PropertyException.class, () -> new NoisyInterlockingCrescentsDataSource(200, 1, -0.1));
        assertThrows(PropertyException.class, () -> new NoisyInterlockingCrescentsDataSource(200, 1, 0.0));

        // Check valid parameters work
        NoisyInterlockingCrescentsDataSource source = new NoisyInterlockingCrescentsDataSource(200, 1, 0.1);
        assertEquals(200, source.examples.size());
        Dataset<Label> dataset = new MutableDataset<>(source);
        Map<String, Long> map = new HashMap<>();
        dataset.getOutputInfo().outputCountsIterable().forEach((p) -> map.put(p.getA(), p.getB()));
        assertEquals(100, map.get("X"));
        assertEquals(100, map.get("O"));
        Helpers.testProvenanceMarshalling(source.getProvenance());
    }

}

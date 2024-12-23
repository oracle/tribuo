/*
 * Copyright (c) 2015, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.sequence;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.DatasetTest;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableDataset;
import org.tribuo.MutableDataset;
import org.tribuo.dataset.DatasetView;
import org.tribuo.dataset.MinimumCardinalityDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.BinaryFeaturesExample;
import org.tribuo.impl.ListExample;
import org.tribuo.protos.core.DatasetProto;
import org.tribuo.protos.core.SequenceDatasetProto;
import org.tribuo.protos.core.SequenceExampleProto;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.util.Merger;

public class SequenceDatasetTest {

    @BeforeAll
    public static void suppressLogging() {
        Logger logger = Logger.getLogger(MinimumCardinalitySequenceDataset.class.getName());
        logger.setLevel(Level.WARNING);
    }

    static MutableSequenceDataset<MockOutput> makeDataset() {
        MutableSequenceDataset<MockOutput> dataset = new MutableSequenceDataset<>(new MockDataSourceProvenance(),
                new MockOutputFactory());

        ListExample<MockOutput> ex1 = new ListExample<>(new MockOutput("green"));
        ex1.add(new Feature("f1", 1.0));
        ex1.add(new Feature("f2", 0.0));
        ex1.add(new Feature("f3", 1.0));
        Example<MockOutput> ex2 = new ListExample<>(new MockOutput("green"));
        ex2.add(new Feature("f1", 1.0));
        ex2.add(new Feature("f2", 0.0));
        ex2.add(new Feature("f3", 1.0));
        SequenceExample<MockOutput> seqEx = new SequenceExample<>(Arrays.asList(ex1, ex2));
        dataset.add(seqEx);

        ex1 = new ListExample<>(new MockOutput("blue"));
        ex1.add(new Feature("f1", 1.0));
        ex1.add(new Feature("f2", 0.0));
        ex1.add(new Feature("f4", 1.0));
        ex2 = new ListExample<>(new MockOutput("green"));
        ex2.add(new Feature("f1", 1.0));
        ex2.add(new Feature("f4", 0.0));
        ex2.add(new Feature("f5", 1.0));
        seqEx = new SequenceExample<>(Arrays.asList(ex1, ex2));
        dataset.add(seqEx);

        return dataset;
    }

    @Test
    public void testBasic() {
        MutableSequenceDataset<MockOutput> dataset = makeDataset();

        FeatureMap infoMap = dataset.getFeatureIDMap();
        assertEquals(4, infoMap.get("f1").getCount());
        assertEquals(0, infoMap.get("f2").getCount());
        assertEquals(2, infoMap.get("f3").getCount());
        // One as non-sparse zeros are ignored
        assertEquals(1, infoMap.get("f4").getCount());
        assertEquals(1, infoMap.get("f5").getCount());
        MutableSequenceDataset<MockOutput> deser = (MutableSequenceDataset<MockOutput>) Helpers.testSequenceDatasetSerialization(dataset);

        ImmutableSequenceDataset<MockOutput> imm = ImmutableSequenceDataset.copyDataset(dataset);
        Helpers.testSequenceDatasetSerialization(imm);

        SequenceDataset<MockOutput> prunedDataset = new MinimumCardinalitySequenceDataset<>(dataset, 2);
        infoMap = prunedDataset.getFeatureIDMap();
        assertEquals(4, infoMap.get("f1").getCount());
        assertNull(infoMap.get("f2"));
        assertEquals(2, infoMap.get("f3").getCount());
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));

        SequenceExample<MockOutput> seqEx = prunedDataset.getExample(1);
        Example<MockOutput> ex2 = seqEx.get(1);
        Feature f1 = ex2.lookup("f1");
        assertEquals(1.0, f1.getValue(), 1e-5);
        assertNull(ex2.lookup("f5"));

        prunedDataset = new MinimumCardinalitySequenceDataset<>(dataset, 3);
        infoMap = prunedDataset.getFeatureIDMap();
        assertEquals(4, infoMap.get("f1").getCount());
        assertNull(infoMap.get("f2"));
        assertNull(infoMap.get("f3"));
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));

        // no examples make it through the pruning
        prunedDataset = new MinimumCardinalitySequenceDataset<>(dataset, 5);
        assertEquals(0, prunedDataset.size());
        infoMap = prunedDataset.getFeatureIDMap();
        assertNull(infoMap.get("f1"));
        assertNull(infoMap.get("f2"));
        assertNull(infoMap.get("f3"));
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));
    }

    @Test
    public void testDuplicateFeatureBug() {
        MockOutputFactory mockOutputFactory = new MockOutputFactory();
        MutableSequenceDataset<MockOutput> dataset = new MutableSequenceDataset<>(
                new SimpleDataSourceProvenance("test dataset", OffsetDateTime.now(), mockOutputFactory),
                mockOutputFactory);
        List<Example<MockOutput>> examples = new ArrayList<>();
        ListExample<MockOutput> example = new ListExample<>(new MockOutput("BUG!"));
        example.add(new Feature("feature1", 1.0));
        example.add(new Feature("feature1", 1.0)); // add duplicate feature
        examples.add(example);
        SequenceExample<MockOutput> seqExample = new SequenceExample<>(examples);
        try {
            dataset.add(seqExample);
            fail("No exception thrown.");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().startsWith("SequenceExample had duplicate features"),
                    "Test for duplicate features");
        }
    }

    @Test
    public void testMinimumCardinality() {
        MutableSequenceDataset<MockOutput> dataset = new MutableSequenceDataset<>(new MockDataSourceProvenance(),
                new MockOutputFactory());

        ListExample<MockOutput> ex1 = createExample("green", "f1");
        ListExample<MockOutput> ex2 = createExample("green", "f1", "f2");
        ListExample<MockOutput> ex3 = createExample("green", "f1", "f2", "f3");
        ListExample<MockOutput> ex4 = createExample("green", "f1", "f2", "f3", "f4");
        SequenceExample<MockOutput> seqEx = new SequenceExample<>(Arrays.asList(ex1, ex2, ex3, ex4));
        dataset.add(seqEx);

        ex1 = createExample("blue", "f1");
        ex2 = createExample("blue", "f1", "f2");
        ex3 = createExample("blue", "f1", "f2", "f3");
        ex4 = createExample("blue", "f1", "f2", "f3", "f4");
        seqEx = new SequenceExample<>(Arrays.asList(ex1, ex2, ex3, ex4));
        dataset.add(seqEx);

        ex1 = createExample("orange", "f5", "f6", "f7");
        seqEx = new SequenceExample<>(Arrays.asList(ex1));
        dataset.add(seqEx);

        FeatureMap infoMap = dataset.getFeatureIDMap();
        assertEquals(8, infoMap.get("f1").getCount());
        assertEquals(6, infoMap.get("f2").getCount());
        assertEquals(4, infoMap.get("f3").getCount());
        assertEquals(2, infoMap.get("f4").getCount());
        assertEquals(1, infoMap.get("f5").getCount());
        assertEquals(1, infoMap.get("f6").getCount());
        assertEquals(1, infoMap.get("f7").getCount());
        assertEquals(3, dataset.size());

        MinimumCardinalitySequenceDataset<MockOutput> minimumCardinalityDataset = new MinimumCardinalitySequenceDataset<>(
                dataset, 3);
        assertEquals(3, minimumCardinalityDataset.getMinCardinality());
        infoMap = minimumCardinalityDataset.getFeatureIDMap();
        assertEquals(8, infoMap.get("f1").getCount());
        assertEquals(6, infoMap.get("f2").getCount());
        assertEquals(4, infoMap.get("f3").getCount());
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));
        assertNull(infoMap.get("f6"));
        assertNull(infoMap.get("f7"));
        assertEquals(2, minimumCardinalityDataset.size());
        MinimumCardinalitySequenceDataset<MockOutput> deser = (MinimumCardinalitySequenceDataset<MockOutput>) Helpers.testSequenceDatasetSerialization(minimumCardinalityDataset);
        assertEquals(minimumCardinalityDataset.getMinCardinality(), deser.getMinCardinality());
    }

    @Test
    public void testBinaryFeatures() {
        MutableSequenceDataset<MockOutput> dataset = new MutableSequenceDataset<>(new MockDataSourceProvenance(), new MockOutputFactory());

        List<MockOutput> outputs = Arrays.asList(new MockOutput("green"), new MockOutput("blue"), new MockOutput("green"), new MockOutput("blue"), new MockOutput("red"));
        List<? extends List<? extends Feature>> features = Arrays.asList(
                Arrays.asList(new Feature("F1", 1.0), new Feature("F2", 1.0), new Feature("F1", 1.0), new Feature("F3", 1.0)),
                Arrays.asList(new Feature("F1", 1.0), new Feature("F2", 1.0)),
                Arrays.asList(new Feature("F1", 1.0), new Feature("F4", 1.0), new Feature("F5", 1.0)),
                Arrays.asList(new Feature("F1", 1.0), new Feature("F5", 2.0)),
                Arrays.asList(new Feature("F1", 1.0), new Feature("F1", 1.0))
                );
        SequenceExample<MockOutput> seqEx = new SequenceExample<>(outputs, features, true);
        seqEx.reduceByName(Merger.max());
        dataset.add(seqEx);

        assertTrue(seqEx.get(0) instanceof BinaryFeaturesExample);
        assertTrue(seqEx.get(1) instanceof BinaryFeaturesExample);
        assertTrue(seqEx.get(2) instanceof BinaryFeaturesExample);
        assertFalse(seqEx.get(3) instanceof BinaryFeaturesExample);
        assertTrue(seqEx.get(4) instanceof BinaryFeaturesExample);

        FeatureMap infoMap = dataset.getFeatureIDMap();
        assertEquals(5, infoMap.get("F1").getCount());
        assertEquals(2, infoMap.get("F2").getCount());
        assertEquals(1, infoMap.get("F3").getCount());
        assertEquals(1, infoMap.get("F4").getCount());
        assertEquals(2, infoMap.get("F5").getCount());
        assertEquals(1, dataset.size());

        MinimumCardinalitySequenceDataset<MockOutput> minimumCardinalityDataset = new MinimumCardinalitySequenceDataset<>(dataset, 3);
        assertEquals(3, minimumCardinalityDataset.getMinCardinality());
        infoMap = minimumCardinalityDataset.getFeatureIDMap();
        assertEquals(5, infoMap.get("F1").getCount());
        assertNull(infoMap.get("F2"));
        assertNull(infoMap.get("F3"));
        assertNull(infoMap.get("F4"));
        assertNull(infoMap.get("F5"));
        assertEquals(1, minimumCardinalityDataset.size());
    }

    @Test
    public void testDense() {
        MockOutputFactory mockFactory = new MockOutputFactory();
        MockDataSourceProvenance mockProvenance = new MockDataSourceProvenance();
        MockOutput mockOutput = new MockOutput("test");

        MutableSequenceDataset<MockOutput> dataset = new MutableSequenceDataset<>(mockProvenance, mockFactory);

        // Empty datasets are dense
        assertTrue(dataset.isDense());

        ArrayExample<MockOutput> first = new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{1,1,1});
        ArrayExample<MockOutput> second = new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{2,2,2});
        ArrayExample<MockOutput> third = new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{3,3,3});

        SequenceExample<MockOutput> denseExample = new SequenceExample<>(Arrays.asList(first,second,third));

        dataset.add(denseExample);
        MutableSequenceDataset<MockOutput> deser = (MutableSequenceDataset<MockOutput>) Helpers.testSequenceDatasetSerialization(dataset);

        // This example is dense
        assertTrue(dataset.isDense());
        assertTrue(deser.isDense());

        first = new ArrayExample<>(mockOutput,new String[]{"a","b","c","d"},new double[]{1,1,1,1});
        second = new ArrayExample<>(mockOutput,new String[]{"a","b","c","d"},new double[]{1,1,1,1});
        third = new ArrayExample<>(mockOutput,new String[]{"a","b","c","d"},new double[]{1,1,1,1});

        SequenceExample<MockOutput> newDenseExample = new SequenceExample<>(Arrays.asList(first,second,third));

        dataset.add(newDenseExample);
        deser = (MutableSequenceDataset<MockOutput>) Helpers.testSequenceDatasetSerialization(dataset);

        // This example is dense, but it makes the previous one not dense as it adds a new feature
        assertFalse(dataset.isDense());
        assertFalse(deser.isDense());

        // flush out the previous test
        dataset.clear();

        first = new ArrayExample<>(mockOutput,new String[]{"a"},new double[]{1});
        second = new ArrayExample<>(mockOutput,new String[]{"a","b"},new double[]{1,1});
        third = new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{1,1,1});
        ArrayExample<MockOutput> fourth = new ArrayExample<>(mockOutput,new String[]{"b","c"},new double[]{1,1});

        SequenceExample<MockOutput> sparseExample = new SequenceExample<>(Arrays.asList(first,second,third,fourth));

        dataset.add(sparseExample);

        // This example is sparse
        assertFalse(dataset.isDense());

        dataset.densify();

        // After densification it should be dense
        assertTrue(dataset.isDense());

        dataset.add(denseExample);

        // should still be dense as they share a feature space
        assertTrue(dataset.isDense());

        dataset.add(newDenseExample);

        // now it's sparse again
        assertFalse(dataset.isDense());

        dataset.densify();

        assertTrue(dataset.isDense());
        Helpers.testSequenceDatasetSerialization(dataset);
    }

    private ListExample<MockOutput> createExample(String outputLabel, String... featureNames) {
        ListExample<MockOutput> example = new ListExample<>(new MockOutput(outputLabel));
        for (String featureName : featureNames) {
            example.add(new Feature(featureName, 1.0));
        }
        return example;
    }

    @Test
    public void test431Protobufs() throws IOException, URISyntaxException {
        MutableSequenceDataset<MockOutput> mutable = makeDataset();
        SequenceExample<MockOutput> example = mutable.getExample(0);
        ImmutableSequenceDataset<MockOutput> immutable = ImmutableSequenceDataset.copyDataset(mutable);
        MinimumCardinalitySequenceDataset<MockOutput> minimum = new MinimumCardinalitySequenceDataset<>(mutable, 2);

        Path examplePath = Paths.get(SequenceDatasetTest.class.getResource("sequence-example-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(examplePath)) {
            SequenceExampleProto proto = SequenceExampleProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            SequenceExample<MockOutput> deserializedExample = (SequenceExample<MockOutput>) SequenceExample.deserialize(proto);
            assertEquals(example,deserializedExample);
        }

        Path mutablePath = Paths.get(SequenceDatasetTest.class.getResource("mutable-sequence-dataset-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(mutablePath)) {
            SequenceDatasetProto proto = SequenceDatasetProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            SequenceDataset<MockOutput> dataset = (SequenceDataset<MockOutput>) SequenceDataset.deserialize(proto);
            assertTrue(Helpers.sequenceDatasetEquals(mutable, dataset));
        }

        Path immutablePath = Paths.get(SequenceDatasetTest.class.getResource("immutable-sequence-dataset-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(immutablePath)) {
            SequenceDatasetProto proto = SequenceDatasetProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            SequenceDataset<MockOutput> dataset = (SequenceDataset<MockOutput>) SequenceDataset.deserialize(proto);
            assertTrue(Helpers.sequenceDatasetEquals(immutable, dataset));
        }

        Path minimumPath = Paths.get(SequenceDatasetTest.class.getResource("minimum-cardinality-sequence-dataset-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(minimumPath)) {
            SequenceDatasetProto proto = SequenceDatasetProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            SequenceDataset<MockOutput> dataset = (SequenceDataset<MockOutput>) SequenceDataset.deserialize(proto);
            assertTrue(Helpers.sequenceDatasetEquals(minimum, dataset));
        }
    }

    public void generateProtobufs() throws IOException {
        MutableSequenceDataset<MockOutput> mutable = makeDataset();
        SequenceExample<MockOutput> example = mutable.getExample(0);
        ImmutableSequenceDataset<MockOutput> immutable = ImmutableSequenceDataset.copyDataset(mutable);
        MinimumCardinalitySequenceDataset<MockOutput> minimum = new MinimumCardinalitySequenceDataset<>(mutable, 2);

        Helpers.writeProtobuf(example, Paths.get("src","test","resources","org","tribuo","sequence","sequence-example-431.tribuo"));
        Helpers.writeProtobuf(mutable, Paths.get("src","test","resources","org","tribuo","sequence","mutable-sequence-dataset-431.tribuo"));
        Helpers.writeProtobuf(immutable, Paths.get("src","test","resources","org","tribuo","sequence","immutable-sequence-dataset-431.tribuo"));
        Helpers.writeProtobuf(minimum, Paths.get("src","test","resources","org","tribuo","sequence","minimum-cardinality-sequence-dataset-431.tribuo"));
    }
}

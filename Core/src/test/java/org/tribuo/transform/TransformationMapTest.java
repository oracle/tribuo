/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.transform;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.protos.core.TestCountTransformerProto;
import org.tribuo.protos.core.TransformerMapProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.transform.transformations.LinearScalingTransformation;
import org.tribuo.transform.transformations.SimpleTransform;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class TransformationMapTest {

    public static MutableDataset<MockOutput> generateDenseDataset() {
        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(), new MockOutputFactory());

        MockOutput output = new MockOutput("UNK");
        String[] featureNames = new String[]{"F0", "F1", "F2", "F3", "F4"};
        ArrayExample<MockOutput> example;

        example = new ArrayExample<>(output, featureNames, new double[]{1, 1, 1, 1, 1});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{1, 2, 3, 4, 5});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{0, 0, 0, 0, 0});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{10, 9, 8, 7, 6});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{2, 2, 2, 2, 2});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{10, 10, 10, 10, 10});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{1, 5, 1, 5, 1});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{5, 1, 5, 1, 5});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{1, 2, 3, 4, 5});
        dataset.add(example);

        return dataset;
    }

    public static MutableDataset<MockOutput> generateSparseDataset() {
        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(), new MockOutputFactory());

        MockOutput output = new MockOutput("UNK");
        String[] featureNames = new String[]{"F0", "F1", "F2", "F3", "F4"};
        ArrayExample<MockOutput> example;

        example = new ArrayExample<>(output, featureNames, new double[]{1, 1, 1, 1, 1});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F0", "F1", "F2", "F3"}, new double[]{1, 2, 3, 4});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F0"}, new double[]{10});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F0", "F2"}, new double[]{1, 1});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F1"}, new double[]{1});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F2"}, new double[]{5});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F1", "F3"}, new double[]{2, 2});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F3"}, new double[]{2});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F3"}, new double[]{4});
        dataset.add(example);

        example = new ArrayExample<>(output, new String[]{"F1", "F2", "F4"}, new double[]{1, 1, 1});
        dataset.add(example);

        return dataset;
    }

    @Test
    public void testRegex() {
        MutableDataset<MockOutput> dataset = generateDenseDataset();

        HashMap<String, List<Transformation>> map = new HashMap<>();

        map.put("F0", Collections.singletonList(SimpleTransform.add(0)));
        map.put("F\\d", Collections.singletonList(SimpleTransform.add(0)));

        TransformationMap t = new TransformationMap(new ArrayList<>(), map);

        try {
            TransformerMap tMap = dataset.createTransformers(t);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            // Pass
        } catch (Exception e) {
            fail("Unexpected exception " + e);
        }
    }

    @Test
    public void testNumTransformers() {
        MutableDataset<MockOutput> dataset = generateDenseDataset();

        HashMap<String, List<Transformation>> map = new HashMap<>();

        map.put("F0", Collections.singletonList(SimpleTransform.add(1)));
        map.put("F[34]", Arrays.asList(SimpleTransform.add(5), SimpleTransform.log()));
        map.put("F1", Arrays.asList(SimpleTransform.add(1), SimpleTransform.mul(5), SimpleTransform.exp()));

        TransformationMap t = new TransformationMap(Collections.singletonList(new LinearScalingTransformation()), map);

        TransformerMap tMap = dataset.createTransformers(t);

        for (Map.Entry<String, List<Transformer>> e : tMap.entrySet()) {
            switch (e.getKey()) {
                case "F0":
                    assertEquals(2, e.getValue().size());
                    break;
                case "F1":
                    assertEquals(4, e.getValue().size());
                    break;
                case "F2":
                    assertEquals(1, e.getValue().size());
                    break;
                case "F3":
                    assertEquals(3, e.getValue().size());
                    break;
                case "F4":
                    assertEquals(3, e.getValue().size());
                    break;
                default:
                    fail("Unknown feature named " + e.getKey());
            }
        }

        TransformerMapProto proto = tMap.serialize();
        TransformerMap deserialized = TransformerMap.deserialize(proto);
        assertEquals(tMap, deserialized);
        assertNotSame(tMap, deserialized);
    }

    @Test
    public void testSparseObservations() {
        MutableDataset<MockOutput> dataset = generateSparseDataset();
        TransformationMap t = new TransformationMap(Collections.singletonList(new CountTransformation()), new HashMap<>());

        TransformerMap tMap = dataset.createTransformers(t, true);

        for (Map.Entry<String, List<Transformer>> e : tMap.entrySet()) {
            CountTransformer countTransformer = (CountTransformer) e.getValue().get(0);
            switch (e.getKey()) {
                case "F0":
                    assertEquals(6, countTransformer.sparseCount);
                    assertEquals(4, countTransformer.count);
                    break;
                case "F1":
                    assertEquals(5, countTransformer.sparseCount);
                    assertEquals(5, countTransformer.count);
                    break;
                case "F2":
                    assertEquals(5, countTransformer.sparseCount);
                    assertEquals(5, countTransformer.count);
                    break;
                case "F3":
                    assertEquals(5, countTransformer.sparseCount);
                    assertEquals(5, countTransformer.count);
                    break;
                case "F4":
                    assertEquals(8, countTransformer.sparseCount);
                    assertEquals(2, countTransformer.count);
                    break;
                default:
                    fail("Unknown feature named " + e.getKey());
            }
        }

        TransformerMapProto proto = tMap.serialize();
        TransformerMap deserialized = TransformerMap.deserialize(proto);
        assertEquals(tMap, deserialized);
        assertNotSame(tMap, deserialized);
    }

    public static class CountTransformation implements Transformation {
        @Override
        public TransformStatistics createStats() {
            return new CountStatistics();
        }

        @Override
        public TransformationProvenance getProvenance() {
            return new CountTransformationProvenance();
        }
    }

    public static class CountTransformationProvenance implements TransformationProvenance {
        public CountTransformationProvenance() {}

        public CountTransformationProvenance(Map<String, Provenance> map) {}

        @Override
        public String getClassName() {
            return CountTransformation.class.getName();
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            return Collections.emptyMap();
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof CountTransformationProvenance;
        }
    }

    private static class CountStatistics implements TransformStatistics {

        public int sparseCount;

        public Map<Double, MutableLong> countMap = new HashMap<>();

        public int count;

        @Override
        public void observeValue(double value) {
            MutableLong l = countMap.computeIfAbsent(value, (k) -> new MutableLong());
            l.increment();
            count++;
        }

        @Override
        @Deprecated
        public void observeSparse() {
            sparseCount++;
        }

        @Override
        public void observeSparse(int count) {
            sparseCount += count;
        }

        @Override
        public Transformer generateTransformer() {
            return new CountTransformer(sparseCount, count, countMap);
        }
    }

    private static class CountTransformer implements Transformer {
        public final int count;
        public final int sparseCount;
        public final Map<Double, MutableLong> countMap;

        public CountTransformer(int sparseCount, int count, Map<Double, MutableLong> countMap) {
            this.count = count;
            this.sparseCount = sparseCount;
            this.countMap = countMap;
        }

        /**
         * Deserialization factory.
         *
         * @param version   The serialized object version.
         * @param className The class name.
         * @param message   The serialized data.
         * @throws InvalidProtocolBufferException If the message is not a {@link TestCountTransformerProto}.
         */
        static CountTransformer deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
            TestCountTransformerProto proto = message.unpack(TestCountTransformerProto.class);
            if (version == 0) {
                Map<Double, MutableLong> countMap = new HashMap<>();
                for (int i = 0; i < proto.getCountMapKeysCount(); i++) {
                    countMap.put(proto.getCountMapKeys(i), new MutableLong(proto.getCountMapValues(i)));
                }
                return new CountTransformer(proto.getSparseCount(), proto.getCount(),
                        countMap);
            } else {
                throw new IllegalArgumentException("Unknown version " + version + " expected {0}");
            }
        }

        @Override
        public double transform(double input) {
            return input;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            CountTransformer that = (CountTransformer) o;
            // MutableLong in OLCUT 5.2.0 doesn't implement equals,
            // so we can't compare countMap with Objects.equals.
            // That'll be fixed in the next OLCUT but for the time being we've got this workaround.
            if (countMap != null ^ that.countMap != null) {
                return false;
            } else if (countMap != null && that.countMap != null) {
                if (countMap.size() != that.countMap.size()) {
                    return false;
                } else {
                    for (Map.Entry<Double, MutableLong> e : countMap.entrySet()) {
                        MutableLong other = that.countMap.get(e.getKey());
                        if ((other == null) || (e.getValue().longValue() != other.longValue())) {
                            return false;
                        }
                    }
                }
            }
            return count == that.count && sparseCount == that.sparseCount;
        }

        @Override
        public int hashCode() {
            return Objects.hash(count, sparseCount, countMap);
        }

        @Override
        public TransformerProto serialize() {
            TransformerProto.Builder protoBuilder = TransformerProto.newBuilder();

            protoBuilder.setVersion(0);
            protoBuilder.setClassName(this.getClass().getName());

            TestCountTransformerProto.Builder transformProto = TestCountTransformerProto.newBuilder()
                    .setCount(count).setSparseCount(sparseCount);
            for (Map.Entry<Double, MutableLong> e : countMap.entrySet()) {
                transformProto.addCountMapKeys(e.getKey());
                transformProto.addCountMapValues(e.getValue().longValue());
            }
            protoBuilder.setSerializedData(Any.pack(transformProto.build()));

            return protoBuilder.build();
        }
    }
}

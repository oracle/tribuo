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

package org.tribuo.transform.transformations;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.tribuo.transform.transformations.SimpleTransform.EPSILON;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;

import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.SimpleTransformProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationMap;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.TransformerMap;
import org.tribuo.transform.transformations.SimpleTransform.Operation;

/**
 *
 */
public class SimpleTransformTest {

    public static MutableDataset<MockOutput> generateDenseDataset() {
        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(),new MockOutputFactory());

        MockOutput output = new MockOutput("UNK");
        String[] featureNames = new String[]{"F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"};
        ArrayExample<MockOutput> example;

        example = new ArrayExample<>(output,featureNames,new double[]{1,1,1,1,1,1,1,1,1,1});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{1,2,3,4,5,6,7,8,9,10});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{10,9,8,7,6,5,4,3,2,1});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{2,2,2,2,2,2,2,2,2,2});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{2,4,6,8,10,12,14,16,18,20});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{1,5,1,5,1,5,1,5,1,5});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{5,1,5,1,5,1,5,1,5,1});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{1,-2,3,-4,5,-6,7,-8,9,-10});
        dataset.add(example);

        return dataset;
    }

    @Test
    public void testAddSub() {
        TransformationMap t = new TransformationMap(Arrays.asList(SimpleTransform.add(10),SimpleTransform.sub(10)),new HashMap<>());
        testSimple(t,a->a);
    }

    @Test
    public void testMulDiv() {
        TransformationMap t = new TransformationMap(Arrays.asList(SimpleTransform.mul(10),SimpleTransform.div(10)),new HashMap<>());
        testSimple(t,a->a);
    }

    @Test
    public void testExpLog() {
        TransformationMap t = new TransformationMap(Arrays.asList(SimpleTransform.exp(),SimpleTransform.log()),new HashMap<>());
        testSimple(t,a->a);
    }

    @Test
    public void testExp() {
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.exp()),new HashMap<>());
        testSimple(t,Math::exp);
    }

    @Test
    public void testLog() {
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.log()),new HashMap<>());
        testSimple(t,Math::log);
    }

    @Test
    public void testAdd() {
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.add(5)),new HashMap<>());
        testSimple(t,(double a) -> a + 5);
    }

    @Test
    public void testSub() {
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.sub(100)),new HashMap<>());
        testSimple(t,(double a) -> a - 100);
    }

    @Test
    public void testMul() {
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.mul(-2)),new HashMap<>());
        testSimple(t,(double a) -> a * -2);
    }

    @Test
    public void testDiv() {
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.div(45)),new HashMap<>());
        testSimple(t,(double a) -> a / 45);
    }

    @Test
    public void testBinarise() {
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.binarise()),new HashMap<>());
        testSimple(t,(double a) -> a < EPSILON ? 0.0 : 1.0);
    }

    public void testSimple(TransformationMap t, DoubleUnaryOperator op) {
        MutableDataset<MockOutput> dataset = generateDenseDataset();

        TransformerMap map = dataset.createTransformers(t);

        Dataset<MockOutput> transformedDataset = map.transformDataset(dataset);

        for (int i = 0; i < dataset.size(); i++) {
            Example<MockOutput> original = dataset.getData().get(i);
            Example<MockOutput> transformed = transformedDataset.getData().get(i);
            assertEquals(original.size(),transformed.size(), "Transformed not the same size as original.");
            Iterator<Feature> origItr = original.iterator();
            Iterator<Feature> transItr = transformed.iterator();

            while (origItr.hasNext() && transItr.hasNext()) {
                Feature origFeature = origItr.next();
                Feature transFeature = transItr.next();
                assertEquals(origFeature.getName(),transFeature.getName());
                assertEquals(op.applyAsDouble(origFeature.getValue()),transFeature.getValue(), 1e-12);
            }
        }

        List<Transformer> f0Transformer = map.get("F0");
        if (f0Transformer.size() == 1) {
            TransformerProto proto = f0Transformer.get(0).serialize();
            Transformer transformer = Transformer.deserialize(proto);

            assertEquals(f0Transformer.get(0), transformer);
            assertNotSame(f0Transformer.get(0), transformer);
        }
    }

    @Test
    public void testThresholdBelow() {
        double min = 0.0;
        double max = Double.POSITIVE_INFINITY;
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.threshold(min, max)),new HashMap<>());
        testThresholding(t,min,max);
    }

    @Test
    public void testThresholdAbove() {
        double min = Double.NEGATIVE_INFINITY;
        double max = 1.0;
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.threshold(min, max)),new HashMap<>());
        testThresholding(t,min,max);
    }

    @Test
    public void testThreshold() {
        double min = 0.0;
        double max = 1.0;
        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.threshold(min, max)),new HashMap<>());
        testThresholding(t,min,max);
    }

    public void testThresholding(TransformationMap t, double min, double max) {
        MutableDataset<MockOutput> dataset = generateDenseDataset();

        TransformerMap map = dataset.createTransformers(t);

        Dataset<MockOutput> transformedDataset = map.transformDataset(dataset);

        for (int i = 0; i < dataset.size(); i++) {
            Example<MockOutput> original = dataset.getData().get(i);
            Example<MockOutput> transformed = transformedDataset.getData().get(i);
            assertEquals(original.size(),transformed.size(), "Transformed not the same size as original.");
            Iterator<Feature> origItr = original.iterator();
            Iterator<Feature> transItr = transformed.iterator();

            while (origItr.hasNext() && transItr.hasNext()) {
                Feature origFeature = origItr.next();
                Feature transFeature = transItr.next();
                assertEquals(origFeature.getName(),transFeature.getName());
                if (origFeature.getValue() < min) {
                    assertEquals(min,transFeature.getValue(),1e-12);
                } else if (origFeature.getValue() > max) {
                    assertEquals(max,transFeature.getValue(),1e-12);
                } else {
                    assertEquals(origFeature.getValue(),transFeature.getValue(),1e-12);
                }
            }
        }
    }

    @Test
    public void testFeatureSpecific() {
        MutableDataset<MockOutput> dataset = generateDenseDataset();

        Map<String,List<Transformation>> map = new HashMap<>();

        map.put("F0",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log()));
        map.put("F2",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log(),SimpleTransform.exp()));
        map.put("F4",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log()));
        map.put("F6",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log(),SimpleTransform.add(5)));
        map.put("F8",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log()));

        TransformationMap t = new TransformationMap(new ArrayList<>(),map);

        TransformerMap transMap = dataset.createTransformers(t);

        Dataset<MockOutput> transformedDataset = transMap.transformDataset(dataset);

        for (int i = 0; i < dataset.size(); i++) {
            Example<MockOutput> original = dataset.getData().get(i);
            Example<MockOutput> transformed = transformedDataset.getData().get(i);
            assertEquals(original.size(),transformed.size(), "Transformed not the same size as original.");
            Iterator<Feature> origItr = original.iterator();
            Iterator<Feature> transItr = transformed.iterator();

            while (origItr.hasNext() && transItr.hasNext()) {
                Feature origFeature = origItr.next();
                Feature transFeature = transItr.next();
                assertEquals(origFeature.getName(),transFeature.getName());
                switch (origFeature.getName()) {
                    case "F0":
                        assertEquals(origFeature.getValue(),transFeature.getValue(), 1e-12);
                        break;
                    case "F2":
                        assertEquals(Math.exp(origFeature.getValue()),transFeature.getValue(), 1e-12);
                        break;
                    case "F4":
                        assertEquals(origFeature.getValue(),transFeature.getValue(), 1e-12);
                        break;
                    case "F6":
                        assertEquals(origFeature.getValue() + 5,transFeature.getValue(), 1e-12);
                        break;
                    case "F8":
                        assertEquals(origFeature.getValue(),transFeature.getValue(), 1e-12);
                        break;
                    default:
                        assertEquals(origFeature.getValue(),transFeature.getValue(), 1e-12);
                        break;
                }
            }
        }
    }

    @Test
    public void testSpecificAndGlobal() {
        MutableDataset<MockOutput> dataset = generateDenseDataset();

        Map<String,List<Transformation>> map = new HashMap<>();

        map.put("F0",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log()));
        map.put("F2",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log(),SimpleTransform.exp()));
        map.put("F4",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log(),SimpleTransform.add(5),SimpleTransform.sub(5)));
        map.put("F6",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log(),SimpleTransform.add(5)));
        map.put("F8",Arrays.asList(SimpleTransform.exp(),SimpleTransform.log()));

        TransformationMap t = new TransformationMap(Collections.singletonList(SimpleTransform.mul(-1)),map);

        TransformerMap transMap = dataset.createTransformers(t);

        Dataset<MockOutput> transformedDataset = transMap.transformDataset(dataset);

        for (int i = 0; i < dataset.size(); i++) {
            Example<MockOutput> original = dataset.getData().get(i);
            Example<MockOutput> transformed = transformedDataset.getData().get(i);
            assertEquals(original.size(),transformed.size(), "Transformed not the same size as original.");
            Iterator<Feature> origItr = original.iterator();
            Iterator<Feature> transItr = transformed.iterator();

            while (origItr.hasNext() && transItr.hasNext()) {
                Feature origFeature = origItr.next();
                Feature transFeature = transItr.next();
                assertEquals(origFeature.getName(),transFeature.getName());
                switch (origFeature.getName()) {
                    case "F2":
                        assertEquals(-(Math.exp(origFeature.getValue())),transFeature.getValue(), 1e-12);
                        break;
                    case "F6":
                        assertEquals(-(origFeature.getValue() + 5),transFeature.getValue(), 1e-12);
                        break;
                    default:
                        assertEquals(-origFeature.getValue(),transFeature.getValue(), 1e-12);
                        break;
                }
            }
        }
    }

    @Test
    void testSerializeSimpleTransform() throws Exception {
        Transformer t = new SimpleTransform(Operation.exp, Math.E, Math.PI);
        TransformerProto tp = t.serialize();
        assertEquals(0, tp.getVersion());
        assertEquals("org.tribuo.transform.transformations.SimpleTransform", tp.getClassName());
        SimpleTransformProto proto = tp.getSerializedData().unpack(SimpleTransformProto.class);
        assertEquals("exp", proto.getOp());
        assertEquals(Math.E, proto.getFirstOperand());
        assertEquals(Math.PI, proto.getSecondOperand());

        Transformer tD = ProtoUtil.deserialize(tp);
        assertEquals(t, tD);
        
        assertEquals(TransformerProto.class, ProtoUtil.getSerializedClass(t));

        
    }    
    

}

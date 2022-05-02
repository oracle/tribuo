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

import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.LinearScalingTransformerProto;
import org.tribuo.protos.core.MeanStdDevTransformerProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationMap;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.TransformerMap;
import org.tribuo.transform.transformations.LinearScalingTransformation.LinearScalingTransformer;
import org.tribuo.transform.transformations.MeanStdDevTransformation.MeanStdDevTransformer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 *
 */
public class LinearScalingTest {

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

        example = new ArrayExample<>(output,featureNames,new double[]{0,0,0,0,0,0,0,0,0,0});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{10,9,8,7,6,5,4,3,2,1});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{2,2,2,2,2,2,2,2,2,2});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{10,10,10,10,10,10,10,10,10,10});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{1,5,1,5,1,5,1,5,1,5});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{5,1,5,1,5,1,5,1,5,1});
        dataset.add(example);

        example = new ArrayExample<>(output,featureNames,new double[]{1,2,3,4,5,6,7,8,9,10});
        dataset.add(example);

        return dataset;
    }

    public void testGlobalLinearScaling(TransformationMap t, double min, double max) {
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
                assertTrue(transFeature.getValue() > min-1e-12);
                assertTrue(transFeature.getValue() < (max+1e-12));
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
    public void testGlobalLinearScalingSimple() {
        TransformationMap t = new TransformationMap(Collections.singletonList(new LinearScalingTransformation()),new HashMap<>());
        testGlobalLinearScaling(t,0,1.0);
    }

    @Test
    public void testGlobalLinearScalingAdd() {
        TransformationMap t = new TransformationMap(Arrays.asList(SimpleTransform.add(5),new LinearScalingTransformation()),new HashMap<>());
        testGlobalLinearScaling(t,0.0,1.0);
    }

    @Test
    public void testGlobalLinearScalingSub() {
        TransformationMap t = new TransformationMap(Arrays.asList(SimpleTransform.sub(5),new LinearScalingTransformation()),new HashMap<>());
        testGlobalLinearScaling(t,0.0,1.0);
    }

    @Test
    public void testGlobalLinearScalingChain() {
        TransformationMap t = new TransformationMap(Arrays.asList(SimpleTransform.add(5),SimpleTransform.sub(10),new LinearScalingTransformation()),new HashMap<>());
        testGlobalLinearScaling(t,0.0,1.0);
    }

    @Test
    public void testGlobalLinearScalingInvertedChain() {
        TransformationMap t = new TransformationMap(Arrays.asList(new LinearScalingTransformation(),SimpleTransform.mul(5),SimpleTransform.sub(2.5)),new HashMap<>());
        testGlobalLinearScaling(t,-2.5,2.5);
    }

    @Test
    public void testGlobalLinearScalingRange() {
        TransformationMap t = new TransformationMap(Collections.singletonList(new LinearScalingTransformation(-5, 5)),new HashMap<>());
        testGlobalLinearScaling(t,-5,5);
    }

    @Test
    public void testGlobalLinearScalingFeatureSpecific() {
        Map<String,List<Transformation>> map = new HashMap<>();

        map.put("F0", Collections.singletonList(SimpleTransform.add(5)));
        map.put("F9", Collections.singletonList(SimpleTransform.div(5)));

        TransformationMap t = new TransformationMap(Collections.singletonList(new LinearScalingTransformation()),map);
        testGlobalLinearScaling(t,0,1);
    }

    @Test
    void testSerialize() throws Exception {
        LinearScalingTransformer lst = new LinearScalingTransformer(Math.E, Math.PI, 0.618033988749, 1.059463094359);
        TransformerProto tp = lst.serialize();
        assertEquals(0, tp.getVersion());
        assertEquals("org.tribuo.transform.transformations.LinearScalingTransformation$LinearScalingTransformer", tp.getClassName());
        LinearScalingTransformerProto proto = tp.getSerializedData().unpack(LinearScalingTransformerProto.class);
        assertEquals(Math.E, proto.getObservedMin());
        assertEquals(Math.PI, proto.getObservedMax());
        assertEquals(0.618033988749, proto.getTargetMin());
        assertEquals(1.059463094359, proto.getTargetMax());

        Transformer tD = ProtoUtil.deserialize(tp);
        assertEquals(lst, tD);
        
        assertEquals(TransformerProto.class, ProtoUtil.getSerializedClass(lst));

    }    

}

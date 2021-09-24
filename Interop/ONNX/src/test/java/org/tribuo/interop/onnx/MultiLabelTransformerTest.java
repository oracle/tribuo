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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.Test;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class MultiLabelTransformerTest {

    private static final MultiLabelFactory factory = new MultiLabelFactory();
    private static final MultiLabelTransformer transformer = new MultiLabelTransformer();

    @Test
    public void multilabelTest() {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            Map<MultiLabel,Integer> outputMapping = new HashMap<>();
            outputMapping.put(new MultiLabel("A"),0);
            outputMapping.put(new MultiLabel("B"),1);
            outputMapping.put(new MultiLabel("C"),2);
            outputMapping.put(new MultiLabel("D"),3);
            ImmutableOutputInfo<MultiLabel> outputInfo = factory.constructInfoForExternalModel(outputMapping);
            OnnxTensor output = OnnxTensor.createTensor(env,new float[][]{{0.1f,0.51f,0.8f,0.0f}});

            // Test transform to output
            MultiLabel m = transformer.transformToOutput(Collections.singletonList(output),outputInfo);
            assertFalse(m.contains("A"));
            assertTrue(m.contains("B"));
            assertTrue(m.contains("C"));
            assertFalse(m.contains("D"));
            assertEquals(0.51f,m.getLabelScore(new Label("B")).getAsDouble());
            assertEquals(0.8f,m.getLabelScore(new Label("C")).getAsDouble());

            // Test transform to prediction
            Prediction<MultiLabel> pred = transformer.transformToPrediction(Collections.singletonList(output),outputInfo,1,new ArrayExample<>(m));
            MultiLabel predOutput = pred.getOutput();
            assertFalse(predOutput.contains("A"));
            assertTrue(predOutput.contains("B"));
            assertTrue(predOutput.contains("C"));
            assertFalse(predOutput.contains("D"));
            assertEquals(0.1f,pred.getOutputScores().get("A").getLabelScore(new Label("A")).getAsDouble());
            assertEquals(0.51f,pred.getOutputScores().get("B").getLabelScore(new Label("B")).getAsDouble());
            assertEquals(0.8f,pred.getOutputScores().get("C").getLabelScore(new Label("C")).getAsDouble());
            assertEquals(0.0f,pred.getOutputScores().get("D").getLabelScore(new Label("D")).getAsDouble());
        } catch (OrtException e) {
            fail(e);
        }
    }

    @Test
    public void multiLabelBatchTest() {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            Map<MultiLabel,Integer> outputMapping = new HashMap<>();
            outputMapping.put(new MultiLabel("A"),0);
            outputMapping.put(new MultiLabel("B"),1);
            outputMapping.put(new MultiLabel("C"),2);
            outputMapping.put(new MultiLabel("D"),3);
            ImmutableOutputInfo<MultiLabel> outputInfo = factory.constructInfoForExternalModel(outputMapping);
            OnnxTensor output = OnnxTensor.createTensor(env,new float[][]{{0.1f,0.51f,0.8f,0.0f},{0.9f,0.1f,0.2f,1.0f},{0.0f,0.0f,0.0f,0.0f}});

            // Test transform to batch output
            List<MultiLabel> m = transformer.transformToBatchOutput(Collections.singletonList(output),outputInfo);
            assertEquals(3,m.size());
            MultiLabel first = m.get(0);
            assertFalse(first.contains("A"));
            assertTrue(first.contains("B"));
            assertTrue(first.contains("C"));
            assertFalse(first.contains("D"));
            assertEquals(0.51f,first.getLabelScore(new Label("B")).getAsDouble());
            assertEquals(0.8f,first.getLabelScore(new Label("C")).getAsDouble());
            MultiLabel second = m.get(1);
            assertTrue(second.contains("A"));
            assertFalse(second.contains("B"));
            assertFalse(second.contains("C"));
            assertTrue(second.contains("D"));
            assertEquals(0.9f,second.getLabelScore(new Label("A")).getAsDouble());
            assertEquals(1.0f,second.getLabelScore(new Label("D")).getAsDouble());
            MultiLabel third = m.get(2);
            assertFalse(third.contains("A"));
            assertFalse(third.contains("B"));
            assertFalse(third.contains("C"));
            assertFalse(third.contains("D"));
            assertEquals(0,third.getLabelSet().size());

            // Test transform to batch prediction
            int[] numValidFeatures = new int[]{1,1,1};
            List<Example<MultiLabel>> examples = new ArrayList<>();
            examples.add(new ArrayExample<>(first));
            examples.add(new ArrayExample<>(second));
            examples.add(new ArrayExample<>(third));
            List<Prediction<MultiLabel>> pred = transformer.transformToBatchPrediction(Collections.singletonList(output),outputInfo,numValidFeatures,examples);
            Prediction<MultiLabel> firstPred = pred.get(0);
            MultiLabel firstOutput = firstPred.getOutput();
            assertFalse(firstOutput.contains("A"));
            assertTrue(firstOutput.contains("B"));
            assertTrue(firstOutput.contains("C"));
            assertFalse(firstOutput.contains("D"));
            assertEquals(0.1f,firstPred.getOutputScores().get("A").getLabelScore(new Label("A")).getAsDouble());
            assertEquals(0.51f,firstPred.getOutputScores().get("B").getLabelScore(new Label("B")).getAsDouble());
            assertEquals(0.8f,firstPred.getOutputScores().get("C").getLabelScore(new Label("C")).getAsDouble());
            assertEquals(0.0f,firstPred.getOutputScores().get("D").getLabelScore(new Label("D")).getAsDouble());
            Prediction<MultiLabel> secondPred = pred.get(1);
            MultiLabel secondOutput = secondPred.getOutput();
            assertTrue(secondOutput.contains("A"));
            assertFalse(secondOutput.contains("B"));
            assertFalse(secondOutput.contains("C"));
            assertTrue(secondOutput.contains("D"));
            assertEquals(0.9f,secondPred.getOutputScores().get("A").getLabelScore(new Label("A")).getAsDouble());
            assertEquals(0.1f,secondPred.getOutputScores().get("B").getLabelScore(new Label("B")).getAsDouble());
            assertEquals(0.2f,secondPred.getOutputScores().get("C").getLabelScore(new Label("C")).getAsDouble());
            assertEquals(1.0f,secondPred.getOutputScores().get("D").getLabelScore(new Label("D")).getAsDouble());
            Prediction<MultiLabel> thirdPred = pred.get(2);
            MultiLabel thirdOutput = thirdPred.getOutput();
            assertFalse(thirdOutput.contains("A"));
            assertFalse(thirdOutput.contains("B"));
            assertFalse(thirdOutput.contains("C"));
            assertFalse(thirdOutput.contains("D"));
            assertEquals(0.0f,thirdPred.getOutputScores().get("A").getLabelScore(new Label("A")).getAsDouble());
            assertEquals(0.0f,thirdPred.getOutputScores().get("B").getLabelScore(new Label("B")).getAsDouble());
            assertEquals(0.0f,thirdPred.getOutputScores().get("C").getLabelScore(new Label("C")).getAsDouble());
            assertEquals(0.0f,thirdPred.getOutputScores().get("D").getLabelScore(new Label("D")).getAsDouble());
        } catch (OrtException e) {
            fail(e);
        }
    }

}

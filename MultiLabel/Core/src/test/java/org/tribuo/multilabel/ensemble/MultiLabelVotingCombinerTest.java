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

package org.tribuo.multilabel.ensemble;

import org.junit.jupiter.api.Test;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class MultiLabelVotingCombinerTest {

    private static final MultiLabelFactory factory = new MultiLabelFactory();
    private static final MultiLabelVotingCombiner votingCombiner = new MultiLabelVotingCombiner();

    private static final Label purpleLbl = new Label("PURPLE");
    private static final Label monkeyLbl = new Label("MONKEY");
    private static final Label dishwasherLbl = new Label("DISHWASHER");
    private static final Label pidgeonLbl = new Label("PIDGEON");
    private static final Label dryerLbl = new Label("DRYER");

    @Test
    public void votingCombinerTest() {
        MultiLabel empty = new MultiLabel(Collections.emptySet());
        MultiLabel purple = new MultiLabel(purpleLbl);
        MultiLabel monkey = new MultiLabel(monkeyLbl);
        MultiLabel dishwasher = new MultiLabel(dishwasherLbl);
        MultiLabel pidgeon = new MultiLabel(pidgeonLbl);
        MultiLabel dryer = new MultiLabel(dryerLbl);
        Set<Label> pmdSet = new HashSet<>();
        pmdSet.add(purpleLbl);
        pmdSet.add(monkeyLbl);
        pmdSet.add(dishwasherLbl);
        MultiLabel pmd = new MultiLabel(pmdSet);

        MutableOutputInfo<MultiLabel> info = factory.generateInfo();
        info.observe(empty);
        info.observe(purple);
        info.observe(monkey);
        info.observe(dishwasher);
        info.observe(pidgeon);
        info.observe(dryer);
        ImmutableOutputInfo<MultiLabel> immutableInfo = info.generateImmutableOutputInfo();

        Example<MultiLabel> testExample = new ArrayExample<>(empty,new String[]{"X_0","X_1","X_2"}, new double[]{1,2,3});
        List<Prediction<MultiLabel>> predictions = new ArrayList<>();

        // Combiner predicts the empty set
        predictions.add(new Prediction<>(empty,3,testExample));
        predictions.add(new Prediction<>(monkey,3,testExample));
        predictions.add(new Prediction<>(purple,3,testExample));
        Prediction<MultiLabel> output = votingCombiner.combine(immutableInfo,predictions);
        assertEquals(empty,output.getOutput());
        assertEquals(5,output.getOutputScores().size());
        assertEquals(1.0/3,output.getOutputScores().get(purpleLbl.getLabel()).getLabelScore(purpleLbl).getAsDouble());
        assertEquals(1.0/3,output.getOutputScores().get(monkeyLbl.getLabel()).getLabelScore(monkeyLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(dishwasherLbl.getLabel()).getLabelScore(dishwasherLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(pidgeonLbl.getLabel()).getLabelScore(pidgeonLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(dryerLbl.getLabel()).getLabelScore(dryerLbl).getAsDouble());
        predictions.clear();

        // Combiner predicts the majority vote for one label
        predictions.add(new Prediction<>(monkey,3,testExample));
        predictions.add(new Prediction<>(monkey,3,testExample));
        predictions.add(new Prediction<>(purple,3,testExample));
        output = votingCombiner.combine(immutableInfo,predictions);
        assertEquals(monkey,output.getOutput());
        assertEquals(5,output.getOutputScores().size());
        assertEquals(1.0/3,output.getOutputScores().get(purpleLbl.getLabel()).getLabelScore(purpleLbl).getAsDouble());
        assertEquals(2.0/3,output.getOutputScores().get(monkeyLbl.getLabel()).getLabelScore(monkeyLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(dishwasherLbl.getLabel()).getLabelScore(dishwasherLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(pidgeonLbl.getLabel()).getLabelScore(pidgeonLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(dryerLbl.getLabel()).getLabelScore(dryerLbl).getAsDouble());
        predictions.clear();

        // Combiner predicts the majority vote for all labels
        predictions.add(new Prediction<>(pmd,3,testExample));
        predictions.add(new Prediction<>(pmd,3,testExample));
        predictions.add(new Prediction<>(purple,3,testExample));
        output = votingCombiner.combine(immutableInfo,predictions);
        assertEquals(pmd,output.getOutput());
        assertEquals(5,output.getOutputScores().size());
        assertEquals(3.0/3,output.getOutputScores().get(purpleLbl.getLabel()).getLabelScore(purpleLbl).getAsDouble());
        assertEquals(2.0/3,output.getOutputScores().get(monkeyLbl.getLabel()).getLabelScore(monkeyLbl).getAsDouble());
        assertEquals(2.0/3,output.getOutputScores().get(dishwasherLbl.getLabel()).getLabelScore(dishwasherLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(pidgeonLbl.getLabel()).getLabelScore(pidgeonLbl).getAsDouble());
        assertEquals(0.0,output.getOutputScores().get(dryerLbl.getLabel()).getLabelScore(dryerLbl).getAsDouble());
        predictions.clear();

        // Combiner predicts the weighted majority vote
        float[] weights = new float[]{0.1f,0.1f,0.0f,2.0f,3.0f};
        predictions.add(new Prediction<>(pmd,3,testExample));
        predictions.add(new Prediction<>(pmd,3,testExample));
        predictions.add(new Prediction<>(purple,3,testExample));
        predictions.add(new Prediction<>(pidgeon,3,testExample));
        predictions.add(new Prediction<>(dryer,3,testExample));
        output = votingCombiner.combine(immutableInfo,predictions,weights);
        assertEquals(dryer,output.getOutput());
        assertEquals(5,output.getOutputScores().size());
        assertEquals(0.2f/5.2f,output.getOutputScores().get(purpleLbl.getLabel()).getLabelScore(purpleLbl).getAsDouble(),1e-6);
        assertEquals(0.2f/5.2f,output.getOutputScores().get(monkeyLbl.getLabel()).getLabelScore(monkeyLbl).getAsDouble(),1e-6);
        assertEquals(0.2f/5.2f,output.getOutputScores().get(dishwasherLbl.getLabel()).getLabelScore(dishwasherLbl).getAsDouble(),1e-6);
        assertEquals(2.0f/5.2f,output.getOutputScores().get(pidgeonLbl.getLabel()).getLabelScore(pidgeonLbl).getAsDouble(),1e-6);
        assertEquals(3.0f/5.2f,output.getOutputScores().get(dryerLbl.getLabel()).getLabelScore(dryerLbl).getAsDouble(),1e-6);
        predictions.clear();

        // Test invalid weights
        float[] newWeights = new float[2];
        predictions.add(new Prediction<>(pmd,3,testExample));
        predictions.add(new Prediction<>(pmd,3,testExample));
        predictions.add(new Prediction<>(purple,3,testExample));
        assertThrows(IllegalArgumentException.class,() -> votingCombiner.combine(immutableInfo,predictions,newWeights));
        predictions.clear();
        predictions.add(new Prediction<>(pmd,3,testExample));
        assertThrows(IllegalArgumentException.class,() -> votingCombiner.combine(immutableInfo,predictions,newWeights));
    }

}

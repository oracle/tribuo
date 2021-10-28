/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.baseline;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.MutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.Map;

/**
 * A trainer for simple baseline classifiers. Use this only for comparison purposes, if you can't beat these
 * baselines, your ML system doesn't work.
 */
public final class DummyClassifierTrainer implements Trainer<Label> {

    /**
     * Types of dummy classifier.
     */
    public enum DummyType {
        /**
         * Samples the label proprotional to the training label frequencies.
         */
        STRATIFIED,
        /**
         * Returns the most frequent training label.
         */
        MOST_FREQUENT,
        /**
         * Samples uniformly from the label domain.
         */
        UNIFORM,
        /**
         * Returns the supplied label for all inputs.
         */
        CONSTANT
    }

    @Config(mandatory = true,description="Type of dummy classifier.")
    private DummyType dummyType;

    @Config(description="Label to use for the constant classifier.")
    private String constantLabel;

    @Config(description="Seed for the RNG.")
    private long seed = 1L;

    private int invocationCount = 0;

    private DummyClassifierTrainer() {}

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if ((dummyType == DummyType.CONSTANT) && (constantLabel == null)) {
            throw new PropertyException("","constantLabel","Please supply a label string when using the type CONSTANT.");
        }
    }

    @Override
    public Model<Label> train(Dataset<Label> examples, Map<String, Provenance> instanceProvenance) {
        return train(examples, instanceProvenance, INCREMENT_INVOCATION_COUNT) ;
    }

    @Override
    public Model<Label> train(Dataset<Label> examples, Map<String, Provenance> instanceProvenance, int invocationCount) {
        if(invocationCount != INCREMENT_INVOCATION_COUNT) {
            this.invocationCount = invocationCount;
        }
        ModelProvenance provenance = new ModelProvenance(DummyClassifierModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), instanceProvenance);
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        this.invocationCount++;
        switch (dummyType) {
            case CONSTANT:
                MutableOutputInfo<Label> labelInfo = examples.getOutputInfo().generateMutableOutputInfo();
                Label constLabel = new Label(constantLabel);
                labelInfo.observe(constLabel);
                return new DummyClassifierModel(provenance,featureMap,labelInfo.generateImmutableOutputInfo(),constLabel);
            case MOST_FREQUENT: {
                ImmutableOutputInfo<Label> immutableLabelInfo = examples.getOutputIDInfo();
                return new DummyClassifierModel(provenance, featureMap, immutableLabelInfo);
            }
            case UNIFORM:
            case STRATIFIED: {
                ImmutableOutputInfo<Label> immutableLabelInfo = examples.getOutputIDInfo();
                return new DummyClassifierModel(provenance, featureMap, immutableLabelInfo, dummyType, seed);
            }
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    @Override
    public int getInvocationCount() {
        return invocationCount;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        this.invocationCount = invocationCount;
    }

    @Override
    public String toString() {
        switch (dummyType) {
            case CONSTANT:
                return "DummyClassifierTrainer(dummyType="+dummyType+",constantLabel="+constantLabel+")";
            case MOST_FREQUENT: {
                return "DummyClassifierTrainer(dummyType="+dummyType+")";
            }
            case UNIFORM:
            case STRATIFIED: {
                return "DummyClassifierTrainer(dummyType="+dummyType+",seed="+seed+")";
            }
            default:
                return "DummyClassifierTrainer(dummyType="+dummyType+")";
        }
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    /**
     * Creates a trainer which creates models which return random labels sampled from the training label distribution.
     * @param seed The RNG seed to use.
     * @return A classification trainer.
     */
    public static DummyClassifierTrainer createStratifiedTrainer(long seed) {
        DummyClassifierTrainer trainer = new DummyClassifierTrainer();
        trainer.dummyType = DummyType.STRATIFIED;
        trainer.seed = seed;
        return trainer;
    }

    /**
     * Creates a trainer which creates models which return a fixed label.
     * @param constantLabel The label to return.
     * @return A classification trainer.
     */
    public static DummyClassifierTrainer createConstantTrainer(String constantLabel) {
        DummyClassifierTrainer trainer = new DummyClassifierTrainer();
        trainer.dummyType = DummyType.CONSTANT;
        trainer.constantLabel = constantLabel;
        return trainer;
    }

    /**
     * Creates a trainer which creates models which return random labels sampled uniformly from the labels seen at training time.
     * @param seed The RNG seed to use.
     * @return A classification trainer.
     */
    public static DummyClassifierTrainer createUniformTrainer(long seed) {
        DummyClassifierTrainer trainer = new DummyClassifierTrainer();
        trainer.dummyType = DummyType.UNIFORM;
        trainer.seed = seed;
        return trainer;
    }

    /**
     * Creates a trainer which creates models which return a fixed label, the one which was most frequent in the training data.
     * @return A classification trainer.
     */
    public static DummyClassifierTrainer createMostFrequentTrainer() {
        DummyClassifierTrainer trainer = new DummyClassifierTrainer();
        trainer.dummyType = DummyType.MOST_FREQUENT;
        return trainer;
    }
}

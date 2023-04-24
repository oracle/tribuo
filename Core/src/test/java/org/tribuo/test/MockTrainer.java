/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.test;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.Map;

/**
 * A trainer for a mocked classifier.
 */
public final class MockTrainer implements Trainer<MockOutput> {

    @Config(mandatory=true, description="MockOutput to use for the constant classifier.")
    private String constantOutput;

    private int invocationCount = 0;

    private MockTrainer() {}

    /**
     * Creates a trainer which creates models which return a fixed label.
     * @param constantOutput The label to return.
     * @return A mocked trainer.
     */
    public MockTrainer(String constantOutput) {
        this.constantOutput = constantOutput;
    }

    @Override
    public Model<MockOutput> train(Dataset<MockOutput> examples, Map<String, Provenance> instanceProvenance) {
        return train(examples, instanceProvenance, INCREMENT_INVOCATION_COUNT) ;
    }

    @Override
    public Model<MockOutput> train(Dataset<MockOutput> examples, Map<String, Provenance> instanceProvenance, int invocationCount) {
        if(invocationCount != INCREMENT_INVOCATION_COUNT) {
            this.invocationCount = invocationCount;
        }
        ModelProvenance provenance = new ModelProvenance(MockModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), instanceProvenance);
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        this.invocationCount++;
        MutableOutputInfo<MockOutput> labelInfo = examples.getOutputInfo().generateMutableOutputInfo();
        MockOutput constMockOutput = new MockOutput(constantOutput);
        labelInfo.observe(constMockOutput);
        return new MockModel("constant-model",provenance,featureMap,labelInfo.generateImmutableOutputInfo(),constMockOutput);
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
        return "MockTrainer(constantOutput="+ constantOutput +")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}

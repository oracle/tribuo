/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.FeatureSetProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

import java.util.ArrayList;
import java.util.List;

public final class MockFeatureSelector implements FeatureSelector<MockOutput> {

    @Config(mandatory=true, description="Feature names to select")
    private List<String> featureNames;

    private List<Double> featureValues;

    /**
     * For OLCUT.
     */
    private MockFeatureSelector() {}

    public MockFeatureSelector(List<String> featureNames) {
        this.featureNames = featureNames;
        postConfig();
    }

    public void postConfig() {
        this.featureValues = new ArrayList<>();
        for (int i = 0; i < featureNames.size(); i++) {
            featureValues.add(Double.NaN);
        }
    }

    @Override
    public boolean isOrdered() {
        return false;
    }

    @Override
    public SelectedFeatureSet select(Dataset<MockOutput> dataset) {
        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(),dataset.getProvenance(),getProvenance());

        return new SelectedFeatureSet(featureNames,featureValues,isOrdered(),provenance);
    }

    @Override
    public FeatureSelectorProvenance getProvenance() {
        return new FeatureSelectorProvenanceImpl(this);
    }
}

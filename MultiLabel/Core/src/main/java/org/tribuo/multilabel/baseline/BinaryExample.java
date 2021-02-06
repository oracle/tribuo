/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.baseline;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.classification.Label;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.util.Merger;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Logger;

/**
 * A simple example which wraps a MultiLabel example, converting the
 * MultiLabel to the presence or absence of a single Label.
 */
class BinaryExample extends Example<Label> {
    private static final Logger logger = Logger.getLogger(BinaryExample.class.getName());

    private final Example<MultiLabel> innerExample;

    /**
     * Creates a BinaryExample, which wraps a MultiLabel example and
     * has a single Label inside.
     * @param innerExample The example to wrap.
     * @param newLabel The new Label.
     */
    BinaryExample(Example<MultiLabel> innerExample, Label newLabel) {
        super(newLabel);
        this.innerExample = innerExample;
    }

    @Override
    protected void sort() {
        logger.warning("Attempting to sort an immutable IndependentMultiLabelTrainer.BinaryExample");
    }

    @Override
    public void add(Feature feature) {
        logger.warning("Attempting to add a feature to an immutable IndependentMultiLabelTrainer.BinaryExample");
    }

    @Override
    public void addAll(Collection<? extends Feature> features) {
        logger.warning("Attempting to add features to an immutable IndependentMultiLabelTrainer.BinaryExample");
    }

    @Override
    public int size() {
        return innerExample.size();
    }

    @Override
    public void removeFeatures(List<Feature> featureList) {
        logger.warning("Attempting to remove features from an immutable IndependentMultiLabelTrainer.BinaryExample");
    }

    @Override
    public void reduceByName(Merger merger) {
        logger.warning("Attempting to reduce features in an immutable IndependentMultiLabelTrainer.BinaryExample");
    }

    @Override
    public boolean validateExample() {
        return innerExample.validateExample();
    }

    @Override
    public boolean isDense(FeatureMap fMap) {
        return innerExample.isDense(fMap);
    }

    @Override
    protected void densify(List<String> featureNames) {
        logger.warning("Attempting to densify an immutable IndependentMultiLabelTrainer.BinaryExample");
    }

    @Override
    public Example<Label> copy() {
        return new BinaryExample(innerExample, output);
    }

    @Override
    public void set(Feature feature) {
        logger.warning("Attempting to mutate a feature to an immutable IndependentMultiLabelTrainer.BinaryExample");
    }

    @Override
    public Iterator<Feature> iterator() {
        return innerExample.iterator();
    }

    @Override
    public void canonicalize(FeatureMap featureMap) {
        logger.finer("Canonicalize is a no-op on BinaryExample.");
    }
}

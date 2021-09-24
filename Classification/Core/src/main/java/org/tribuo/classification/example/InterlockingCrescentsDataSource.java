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

package org.tribuo.classification.example;

import org.tribuo.Example;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;

import java.util.ArrayList;
import java.util.List;

/**
 * A data source of two interleaved half circles.
 * <p>
 * Also known as the two moons dataset.
 */
public final class InterlockingCrescentsDataSource extends DemoLabelDataSource {

    /**
     * For OLCUT.
     */
    private InterlockingCrescentsDataSource() {
        super();
    }

    /**
     * Constructs an interlocking crescents data source.
     *
     * @param numSamples The number of samples to generate.
     */
    public InterlockingCrescentsDataSource(int numSamples) {
        super(numSamples, Trainer.DEFAULT_SEED);
        postConfig();
    }

    @Override
    protected List<Example<Label>> generate() {
        List<Example<Label>> list = new ArrayList<>();

        for (int i = 0; i < numSamples / 2; i++) {
            double[] values = new double[2];
            values[0] = Math.cos(Math.PI * ((double) i) / ((numSamples / 2) - 1));
            values[1] = Math.sin(Math.PI * ((double) i) / ((numSamples / 2) - 1));
            list.add(new ArrayExample<>(FIRST_CLASS, FEATURE_NAMES, values));
        }

        for (int i = numSamples / 2; i < numSamples; i++) {
            int j = i - numSamples / 2;
            double[] values = new double[2];

            values[0] = 1 - Math.cos(Math.PI * ((double) j) / ((numSamples / 2) - 1));
            values[1] = 0.5 - Math.sin(Math.PI * ((double) j) / ((numSamples / 2) - 1));
            list.add(new ArrayExample<>(SECOND_CLASS, FEATURE_NAMES, values));
        }

        return list;
    }

    @Override
    public String toString() {
        return "InterlockingCrescents(numSamples=" + numSamples + ")";
    }
}

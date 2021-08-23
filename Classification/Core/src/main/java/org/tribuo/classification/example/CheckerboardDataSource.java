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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.Example;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;

import java.util.ArrayList;
import java.util.List;

/**
 * Creates a data source using a 2d checkerboard of alternating classes.
 */
public final class CheckerboardDataSource extends DemoLabelDataSource {

    @Config(description = "The number of squares on each side.")
    private int numSquares = 5;

    @Config(description = "The minimum feature value.")
    private double min = 0.0;

    @Config(description = "The maximum feature value.")
    private double max = 10.0;

    private double range;

    private double tileWidth;

    /**
     * For OLCUT.
     */
    private CheckerboardDataSource() {
        super();
    }

    /**
     * Creates a checkboard with the required number of squares per dimension, where each feature value lies between min and max.
     *
     * @param numSamples The number of samples to generate.
     * @param seed       The RNG seed.
     * @param numSquares The number of squares.
     * @param min        The minimum feature value.
     * @param max        The maximum feature value.
     */
    public CheckerboardDataSource(int numSamples, long seed, int numSquares, double min, double max) {
        super(numSamples, seed);
        this.numSquares = numSquares;
        this.min = min;
        this.max = max;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (max <= min) {
            throw new PropertyException("", "min", "min must be strictly less than max, min = " + min + ", max = " + max);
        }
        if (numSquares < 2) {
            throw new PropertyException("", "numSquares", "numSquares must be 2 or greater, found " + numSquares);
        }
        range = Math.abs(max - min);
        tileWidth = range / numSquares;
        super.postConfig();
    }

    @Override
    protected List<Example<Label>> generate() {
        List<Example<Label>> list = new ArrayList<>();

        for (int i = 0; i < numSamples; i++) {
            double[] values = new double[2];
            values[0] = (rng.nextDouble() * range);
            values[1] = (rng.nextDouble() * range);

            int modX1 = ((int) Math.floor(values[0] / tileWidth)) % 2;
            int modX2 = ((int) Math.floor(values[1] / tileWidth)) % 2;

            Label label;
            if (modX1 == modX2) {
                label = FIRST_CLASS;
            } else {
                label = SECOND_CLASS;
            }

            // Update the minimums after computing the label so we don't have to
            // deal with tricky negative issues interacting with Math.floor().
            values[0] += min;
            values[1] += min;

            list.add(new ArrayExample<>(label, FEATURE_NAMES, values));
        }

        return list;
    }

    @Override
    public String toString() {
        return "Checkerboard(numSamples=" + numSamples + ",seed=" + seed + ",numSquares=" + numSquares + ",min=" + min + ",max=" + max + ')';
    }
}

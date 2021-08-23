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
 * A data source for two concentric circles, one per class.
 */
public final class ConcentricCirclesDataSource extends DemoLabelDataSource {

    @Config(description = "The radius of the outer circle.")
    private double radius = 2;

    @Config(description = "The proportion of the circle radius that forms class one.")
    private double classProportion = 0.5;

    /**
     * For OLCUT.
     */
    private ConcentricCirclesDataSource() {
        super();
    }

    /**
     * Constructs a data source for two concentric circles, one per class.
     *
     * @param numSamples      The number of samples to generate.
     * @param seed            The RNG seed.
     * @param radius          The radius of the outer circle.
     * @param classProportion The proportion of the circle area that forms class 1.
     */
    public ConcentricCirclesDataSource(int numSamples, long seed, double radius, double classProportion) {
        super(numSamples, seed);
        this.radius = radius;
        this.classProportion = classProportion;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if ((classProportion <= 0.0) || (classProportion >= 1.0)) {
            throw new PropertyException("", "classProportion", "Class proportion must be between zero and one, found " + classProportion);
        }
        if (radius <= 0) {
            throw new PropertyException("", "radius", "Radius must be positive, found " + radius);
        }
        super.postConfig();
    }

    @Override
    protected List<Example<Label>> generate() {
        List<Example<Label>> list = new ArrayList<>();

        for (int i = 0; i < numSamples; i++) {
            double rotation = rng.nextDouble() * 2 * Math.PI;
            double distance = Math.sqrt(rng.nextDouble()) * radius;
            double[] values = new double[2];
            values[0] = distance * Math.cos(rotation);
            values[1] = distance * Math.sin(rotation);

            double labelDistance = (values[0] * values[0]) + (values[1] * values[1]);
            Label label;
            if (labelDistance < classProportion * radius * radius) {
                label = FIRST_CLASS;
            } else {
                label = SECOND_CLASS;
            }

            list.add(new ArrayExample<>(label, FEATURE_NAMES, values));
        }

        return list;
    }

    @Override
    public String toString() {
        return "ConcentricCircles(numSamples=" + numSamples + ",seed=" + seed + ",radius=" + radius + ",classProportion=" + classProportion + ")";
    }
}

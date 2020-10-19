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

package org.tribuo.data.columnar.processors.response;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

/**
 * A quartile to split data into 4 chunks.
 */
public class Quartile implements Configurable, Provenancable<ConfiguredObjectProvenance> {

    @Config(mandatory = true,description="The median value.")
    private double median;

    @Config(mandatory = true,description="The lower quartile value.")
    private double lowerMedian;

    @Config(mandatory = true,description="The upper quartile value.")
    private double upperMedian;

    /**
     * Constructs a quartile with the specified values.
     * @param median The median.
     * @param lowerMedian The lower quartile.
     * @param upperMedian The upper quartile.
     */
    public Quartile(double median, double lowerMedian, double upperMedian) {
        this.median = median;
        this.lowerMedian = lowerMedian;
        this.upperMedian = upperMedian;
    }

    /**
     * For olcut.
     */
    private Quartile() {}

    /**
     * Returns the median value.
     * @return The median.
     */
    public double getMedian() {
        return median;
    }

    /**
     * Returns the lower quartile value.
     * @return The lower quartile value.
     */
    public double getLowerMedian() {
        return lowerMedian;
    }

    /**
     * The upper quartile value.
     * @return The upper quartile value.
     */
    public double getUpperMedian() {
        return upperMedian;
    }

    @Override
    public String toString() {
        return "Quartile(lowerMedian="+lowerMedian+",median="+median+",upperMedian="+upperMedian+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"Quartile");
    }
}

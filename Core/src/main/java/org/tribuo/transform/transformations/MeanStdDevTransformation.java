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

package org.tribuo.transform.transformations;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;
import org.tribuo.transform.TransformStatistics;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationProvenance;
import org.tribuo.transform.Transformer;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * A Transformation which takes an observed distribution and rescales
 * it so it has the desired mean and standard deviation.
 * <p>
 * Checks to see that the requested standard deviation is
 * positive, throws IllegalArgumentException otherwise.
 */
public final class MeanStdDevTransformation implements Transformation {

    private static final String TARGET_MEAN = "targetMean";
    private static final String TARGET_STDDEV = "targetStdDev";

    @Config(mandatory = true,description="Mean value after transformation.")
    private double targetMean = 0.0;

    @Config(mandatory = true,description="Standard deviation after transformation.")
    private double targetStdDev = 1.0;

    private MeanStdDevTransformationProvenance provenance;

    /**
     * Defaults to zero mean, one std dev.
     */
    public MeanStdDevTransformation() { }

    public MeanStdDevTransformation(double targetMean, double targetStdDev) {
        this.targetMean = targetMean;
        this.targetStdDev = targetStdDev;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (targetStdDev < SimpleTransform.EPSILON) {
            throw new IllegalArgumentException("Target standard deviation must be positive, found " + targetStdDev);
        }
    }

    @Override
    public TransformStatistics createStats() {
        return new MeanStdDevStatistics(targetMean,targetStdDev);
    }

    @Override
    public TransformationProvenance getProvenance() {
        if (provenance == null) {
            provenance = new MeanStdDevTransformationProvenance(this);
        }
        return provenance;
    }

    /**
     * Provenance for {@link MeanStdDevTransformation}.
     */
    public final static class MeanStdDevTransformationProvenance implements TransformationProvenance {
        private static final long serialVersionUID = 1L;

        private final DoubleProvenance targetMean;
        private final DoubleProvenance targetStdDev;

        MeanStdDevTransformationProvenance(MeanStdDevTransformation host) {
            this.targetMean = new DoubleProvenance(TARGET_MEAN, host.targetMean);
            this.targetStdDev = new DoubleProvenance(TARGET_STDDEV, host.targetStdDev);
        }

        public MeanStdDevTransformationProvenance(Map<String, Provenance> map) {
            targetMean = ObjectProvenance.checkAndExtractProvenance(map, TARGET_MEAN, DoubleProvenance.class, MeanStdDevTransformationProvenance.class.getSimpleName());
            targetStdDev = ObjectProvenance.checkAndExtractProvenance(map, TARGET_STDDEV, DoubleProvenance.class, MeanStdDevTransformationProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return MeanStdDevTransformation.class.getName();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof MeanStdDevTransformationProvenance)) return false;
            MeanStdDevTransformationProvenance pairs = (MeanStdDevTransformationProvenance) o;
            return targetMean.equals(pairs.targetMean) &&
                    targetStdDev.equals(pairs.targetStdDev);
        }

        @Override
        public int hashCode() {
            return Objects.hash(targetMean, targetStdDev);
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            Map<String,Provenance> map = new HashMap<>();
            map.put(TARGET_MEAN,targetMean);
            map.put(TARGET_STDDEV,targetStdDev);
            return Collections.unmodifiableMap(map);
        }
    }

    @Override
    public String toString() {
        return "MeanStdDevTransformation(targetMean="+targetMean+",targetStdDev="+targetStdDev+")";
    }

    private static class MeanStdDevStatistics implements TransformStatistics {
        private static final Logger logger = Logger.getLogger(MeanStdDevStatistics.class.getName());

        private final double targetMean;
        private final double targetStdDev;

        private double mean = 0;
        private double sumSquares = 0;
        private long count = 0;

        public MeanStdDevStatistics(double targetMean, double targetStdDev) {
            this.targetMean = targetMean;
            this.targetStdDev = targetStdDev;
        }

        @Override
        public void observeValue(double value) {
            count++;
            double delta = value - mean;
            mean += delta / count;
            double delta2 = value - mean;
            sumSquares += delta * delta2;
        }

        @Override
        @Deprecated
        public void observeSparse() {
            observeValue(0.0);
        }

        @Override
        public void observeSparse(int sparseCount) {
            count += sparseCount;
            double delta = -mean;
            mean += delta; // implicit zero for delta = 0 - mean;
            double delta2 = -mean;
            sumSquares += sparseCount * (delta * delta2);
        }

        @Override
        public Transformer generateTransformer() {
            if (sumSquares == 0.0) {
                logger.info("Only observed a single value (" + mean + ") when building a MeanStdDevTransformation");
            }
            return new MeanStdDevTransformer(mean,Math.sqrt(sumSquares/(count-1)),targetMean,targetStdDev);
        }

        @Override
        public String toString() {
            return "MeanStdDevStatistics(mean="+mean
                    +",sumSquares="+sumSquares+",count="+count
                    +"targetMean="+targetMean+",targetStdDev="+targetStdDev+")";
        }
    }

    private static class MeanStdDevTransformer implements Transformer {
        private static final long serialVersionUID = 1L;

        private final double observedMean;
        private final double observedStdDev;
        private final double targetMean;
        private final double targetStdDev;

        public MeanStdDevTransformer(double observedMean, double observedStdDev, double targetMean, double targetStdDev) {
            this.observedMean = observedMean;
            this.observedStdDev = observedStdDev;
            this.targetMean = targetMean;
            this.targetStdDev = targetStdDev;
        }

        @Override
        public double transform(double input) {
            return (((input - observedMean) / observedStdDev) * targetStdDev) + targetMean;
        }

        @Override
        public String toString() {
            return "MeanStdDevTransformer(observedMean="+observedMean+",observedStdDev="+observedStdDev+",targetMean="+targetMean+",targetStdDev="+targetStdDev+")";
        }
    }
}


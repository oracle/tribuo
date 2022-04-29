/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.core.MeanStdDevTransformerProto;
import org.tribuo.transform.TransformStatistics;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationProvenance;
import org.tribuo.transform.Transformer;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;

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

    /**
     * Constructs a MeanStdDevTransformation targetting the specified mean and standard deviation.
     * @param targetMean The target mean.
     * @param targetStdDev The target standard deviation.
     */
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

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
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

    @ProtoSerializableClass(serializedDataClass = MeanStdDevTransformerProto.class)
    static final class MeanStdDevTransformer implements Transformer {
        private static final long serialVersionUID = 1L;

        @ProtoSerializableField
        private final double observedMean;
        @ProtoSerializableField
        private final double observedStdDev;
        @ProtoSerializableField
        private final double targetMean;
        @ProtoSerializableField
        private final double targetStdDev;

        MeanStdDevTransformer(double observedMean, double observedStdDev, double targetMean, double targetStdDev) {
            if ((observedStdDev < 0) || (targetStdDev < 0)) {
                throw new IllegalArgumentException("Standard deviations must be non-negative.");
            }
            this.observedMean = observedMean;
            this.observedStdDev = observedStdDev;
            this.targetMean = targetMean;
            this.targetStdDev = targetStdDev;
        }

        /**
         * Deserialization factory.
         * @param version The serialized object version.
         * @param className The class name.
         * @param message The serialized data.
         * @throws InvalidProtocolBufferException If the message is not a {@link MeanStdDevTransformerProto}.
         */
        static MeanStdDevTransformer deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
            MeanStdDevTransformerProto proto = message.unpack(MeanStdDevTransformerProto.class);
            if (version == 0) {
                return new MeanStdDevTransformer(proto.getObservedMean(),proto.getObservedStdDev(),
                        proto.getTargetMean(),proto.getTargetStdDev());
            } else {
                throw new IllegalArgumentException("Unknown version " + version + " expected {0}");
            }
        }

        @Override
        public double transform(double input) {
            return (((input - observedMean) / observedStdDev) * targetStdDev) + targetMean;
        }


        @Override
        public String toString() {
            return "MeanStdDevTransformer(observedMean="+observedMean+",observedStdDev="+observedStdDev+",targetMean="+targetMean+",targetStdDev="+targetStdDev+")";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            MeanStdDevTransformer that = (MeanStdDevTransformer) o;
            return Double.compare(that.observedMean, observedMean) == 0 && Double.compare(that.observedStdDev, observedStdDev) == 0 && Double.compare(that.targetMean, targetMean) == 0 && Double.compare(that.targetStdDev, targetStdDev) == 0;
        }

        @Override
        public int hashCode() {
            return Objects.hash(observedMean, observedStdDev, targetMean, targetStdDev);
        }
    }
}


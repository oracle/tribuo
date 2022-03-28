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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;
import org.tribuo.protos.core.LinearScalingTransformerProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.transform.TransformStatistics;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationProvenance;
import org.tribuo.transform.Transformer;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A Transformation which takes an observed distribution and rescales
 * it so all values are between the desired min and max. The scaling
 * is linear.
 * <p>
 * Values outside the observed range are clamped to the desired
 * min or max.
 */
public final class LinearScalingTransformation implements Transformation {

    private static final String TARGET_MIN = "targetMin";
    private static final String TARGET_MAX = "targetMax";

    @Config(mandatory = true,description="Minimum value after transformation.")
    private double targetMin = 0.0;

    @Config(mandatory = true,description="Maximum value after transformation.")
    private double targetMax = 1.0;

    private TransformationProvenance provenance;

    /**
     * Defaults to zero - one.
     */
    public LinearScalingTransformation() { }

    /**
     * Constructs a LinearScalingTransformation which puts feature values into the specified range.
     * @param targetMin The new minimum feature value.
     * @param targetMax The new maximum feature value.
     */
    public LinearScalingTransformation(double targetMin, double targetMax) {
        this.targetMin = targetMin;
        this.targetMax = targetMax;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (targetMax < targetMin) {
            throw new IllegalArgumentException("Range must be positive, min = " + targetMin + ", max = " + targetMax);
        }
    }

    @Override
    public TransformStatistics createStats() {
        return new LinearScalingStatistics(targetMin, targetMax);
    }

    @Override
    public synchronized TransformationProvenance getProvenance() {
        if (provenance == null) {
            provenance = new LinearScalingTransformationProvenance(this);
        }
        return provenance;
    }

    /**
     * Provenance for {@link LinearScalingTransformation}.
     */
    public final static class LinearScalingTransformationProvenance implements TransformationProvenance {
        private static final long serialVersionUID = 1L;

        private final DoubleProvenance targetMin;
        private final DoubleProvenance targetMax;

        LinearScalingTransformationProvenance(LinearScalingTransformation host) {
            this.targetMin = new DoubleProvenance(TARGET_MIN,host.targetMin);
            this.targetMax = new DoubleProvenance(TARGET_MAX,host.targetMax);
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public LinearScalingTransformationProvenance(Map<String,Provenance> map) {
            targetMin = ObjectProvenance.checkAndExtractProvenance(map,TARGET_MIN,DoubleProvenance.class,LinearScalingTransformationProvenance.class.getSimpleName());
            targetMax = ObjectProvenance.checkAndExtractProvenance(map,TARGET_MAX,DoubleProvenance.class,LinearScalingTransformationProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return LinearScalingTransformation.class.getName();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof LinearScalingTransformationProvenance)) return false;
            LinearScalingTransformationProvenance pairs = (LinearScalingTransformationProvenance) o;
            return targetMin.equals(pairs.targetMin) &&
                    targetMax.equals(pairs.targetMax);
        }

        @Override
        public int hashCode() {
            return Objects.hash(targetMin, targetMax);
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            Map<String,Provenance> map = new HashMap<>();
            map.put(TARGET_MIN,targetMin);
            map.put(TARGET_MAX,targetMax);
            return Collections.unmodifiableMap(map);
        }
    }

    @Override
    public String toString() {
        return "LinearScalingTransformation(targetMin=" + targetMin + ",targetMax=" + targetMax + ")";
    }

    private static class LinearScalingStatistics implements TransformStatistics {

        private final double targetMin;
        private final double targetMax;

        private double min = Double.POSITIVE_INFINITY;
        private double max = Double.NEGATIVE_INFINITY;

        public LinearScalingStatistics(double targetMin, double targetMax) {
            this.targetMin = targetMin;
            this.targetMax = targetMax;
        }

        @Override
        public void observeValue(double value) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }

        @Override
        @Deprecated
        public void observeSparse() {
            observeValue(0.0);
        }

        @Override
        public void observeSparse(int count) {
            // This just tracks max and min, so seeing many 0.0 is the same as one 0.0.
            observeValue(0.0);
        }

        @Override
        public Transformer generateTransformer() {
            return new LinearScalingTransformer(min, max, targetMin, targetMax);
        }

        @Override
        public String toString() {
            return "LinearScalingStatistics(min="+min+",max="+max
                    +",targetMin="+targetMin+",targetMax="+targetMax+")";
        }
    }

    private static final class LinearScalingTransformer implements Transformer {
        private static final long serialVersionUID = 1L;

        private final double observedMin;
        private final double observedMax;
        private final double targetMin;
        private final double targetMax;
        private final double scalingFactor;
        private final boolean constant;

        public LinearScalingTransformer(double observedMin, double observedMax, double targetMin, double targetMax) {
            if ((observedMin > observedMax) || (targetMin > targetMax)) {
                throw new IllegalArgumentException("observedMin and targetMin must be less than observedMax and targetMax respectively");
            }
            this.observedMin = observedMin;
            this.observedMax = observedMax;
            this.targetMin = targetMin;
            this.targetMax = targetMax;
            double observedRange = observedMax - observedMin;
            this.constant = (observedRange == 0.0);
            double targetRange = targetMax - targetMin;
            this.scalingFactor = targetRange / observedRange;
        }

        /**
         * Deserialization factory.
         * @param version The serialized object version.
         * @param className The class name.
         * @param message The serialized data.
         * @throws InvalidProtocolBufferException If the message is not a {@link LinearScalingTransformerProto}.
         */
        public static LinearScalingTransformer deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
            LinearScalingTransformerProto proto = message.unpack(LinearScalingTransformerProto.class);
            if (version == 0) {
                return new LinearScalingTransformer(proto.getObservedMin(),proto.getObservedMax(),
                        proto.getTargetMin(),proto.getTargetMax());
            } else {
                throw new IllegalArgumentException("Unknown version " + version + " expected {0}");
            }
        }

        @Override
        public double transform(double input) {
            if (constant) {
                return (targetMax - targetMin) / 2.0;
            } else if (input < observedMin) {
                // If outside observed range, clamp to min or max.
                return targetMin;
            } else if (input > observedMax) {
                return targetMax;
            } else {
                return ((input - observedMin) * scalingFactor) + targetMin;
            }
        }

        @Override
        public TransformerProto serialize() {
            TransformerProto.Builder protoBuilder = TransformerProto.newBuilder();

            protoBuilder.setVersion(0);
            protoBuilder.setClassName(this.getClass().getName());

            LinearScalingTransformerProto transformProto = LinearScalingTransformerProto.newBuilder()
                    .setObservedMin(observedMin).setObservedMax(observedMax)
                    .setTargetMin(targetMin).setTargetMax(targetMax).build();
            protoBuilder.setSerializedData(Any.pack(transformProto));

            return protoBuilder.build();
        }

        @Override
        public String toString() {
            return "LinearScalingTransformer(observedMin="+observedMin+",observedMax="+observedMax+",targetMin="+targetMin+",targetMax="+targetMax+")";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            LinearScalingTransformer that = (LinearScalingTransformer) o;
            return Double.compare(that.observedMin, observedMin) == 0 && Double.compare(that.observedMax, observedMax) == 0 && Double.compare(that.targetMin, targetMin) == 0 && Double.compare(that.targetMax, targetMax) == 0 && Double.compare(that.scalingFactor, scalingFactor) == 0 && constant == that.constant;
        }

        @Override
        public int hashCode() {
            return Objects.hash(observedMin, observedMax, targetMin, targetMax, scalingFactor, constant);
        }
    }
}

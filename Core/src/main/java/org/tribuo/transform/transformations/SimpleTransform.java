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

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.SimpleTransformProto;
import org.tribuo.protos.core.TransformerProto;
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
import com.oracle.labs.mlrg.olcut.provenance.primitives.EnumProvenance;

/**
 * This is used for stateless functions such as exp, log, addition or multiplication by a constant.
 * <p>
 * It's a Transformation, Transformer and TransformStatistics as it has
 * no data dependent state. This means a single Transformer can be
 * used for every feature in a dataset.
 * <p>
 * Wraps a {@link DoubleUnaryOperator} which actually performs the
 * transformation.
 */
@ProtoSerializableClass(serializedDataClass = SimpleTransformProto.class)
public final class SimpleTransform implements Transformer, Transformation, TransformStatistics {
    private static final long serialVersionUID = 1L;

    private static final String OP = "op";
    private static final String OPERAND = "operand";
    private static final String SECOND_OPERAND = "secondOperand";

    /**
     * Epsilon for determining when two double values are the same.
     */
    public static final double EPSILON = 1e-12;

    /**
     * Operations understood by this Transformation.
     */
    public enum Operation {
        /**
         * Exponentiates the inputs
         */
        exp,
        /**
         * Logs the inputs (base_e)
         */
        log,
        /**
         * Adds the specified constant.
         */
        add,
        /**
         * Subtracts the specified constant.
         */
        sub,
        /**
         * Multiplies by the specified constant.
         */
        mul,
        /**
         * Divides by the specified constant.
         */
        div,
        /**
         * Binarises the output around 1.0.
         */
        binarise,
        /**
         * Min and max thresholds applied to the input.
         */
        threshold
    }

    @ProtoSerializableField
    @Config(mandatory = true,description="Type of the simple transformation.")
    private Operation op;

    @ProtoSerializableField(name="firstOperand")
    @Config(description="Operand (if required).")
    private double operand = Double.NaN;

    @Config(description="Second operand (if required).")
    @ProtoSerializableField
    private double secondOperand = Double.NaN;

    private SerializableDoubleUnaryOperator operation;

    private transient TransformationProvenance provenance;

    /**
     * For OLCUT.
     */
    private SimpleTransform() {}

    SimpleTransform(Operation op, double operand, double secondOperand) {
        this.op = op;
        this.operand = operand;
        this.secondOperand = secondOperand;
        postConfig();
    }

    private SimpleTransform(Operation op, double operand) {
        this.op = op;
        this.operand = operand;
        postConfig();
    }

    private SimpleTransform(Operation op) {
        this.op = op;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        switch (op) {
            case exp:
                operation = Math::exp;
                break;
            case log:
                operation = Math::log;
                break;
            case add:
                if (Double.isNaN(operand)) {
                    throw new IllegalArgumentException("operand must not be NaN");
                }
                operation = (double input) -> input + operand;
                break;
            case sub:
                if (Double.isNaN(operand)) {
                    throw new IllegalArgumentException("operand must not be NaN");
                }
                operation = (double input) -> input - operand;
                break;
            case mul:
                if (Double.isNaN(operand)) {
                    throw new IllegalArgumentException("operand must not be NaN");
                }
                operation = (double input) -> input * operand;
                break;
            case div:
                if (Double.isNaN(operand)) {
                    throw new IllegalArgumentException("operand must not be NaN");
                }
                operation = (double input) -> input / operand;
                break;
            case binarise:
                operation = (double input) -> input < EPSILON ? 0.0 : 1.0;
                break;
            case threshold:
                if (operand > secondOperand) {
                    throw new IllegalArgumentException("Min must be greater than max, min = " + operand + ", max = " + secondOperand);
                } else if (Double.isNaN(operand) || Double.isNaN(secondOperand)) {
                    throw new IllegalArgumentException("min and/or max must not be NaN");
                }
                operation = (double input) -> { if (input < operand) { return operand; } else if (input > secondOperand) { return secondOperand; } else { return input; } };
                break;
            default:
                throw new IllegalArgumentException("Operation " + op + " is unknown");
        }
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the message is not a {@link SimpleTransformProto}.
     */
    static SimpleTransform deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        SimpleTransformProto proto = message.unpack(SimpleTransformProto.class);
        if (version == 0) {
            return new SimpleTransform(Operation.valueOf(proto.getOp()), proto.getFirstOperand(), proto.getSecondOperand());
        } else {
            throw new IllegalArgumentException("Unknown version " + version + " expected {0}");
        }
    }

    @Override
    public TransformationProvenance getProvenance() {
        if (provenance == null) {
            provenance = new SimpleTransformProvenance(this);
        }
        return provenance;
    }

    @Override
    public TransformerProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SimpleTransform that = (SimpleTransform) o;
        return Double.compare(that.operand, operand) == 0 && Double.compare(that.secondOperand, secondOperand) == 0 && op == that.op;
    }

    @Override
    public int hashCode() {
        return Objects.hash(op, operand, secondOperand);
    }

    /**
     * Provenance for {@link SimpleTransform}.
     */
    public final static class SimpleTransformProvenance implements TransformationProvenance {
        private static final long serialVersionUID = 1L;

        private final EnumProvenance<Operation> op;
        private final DoubleProvenance operand;
        private final DoubleProvenance secondOperand;

        SimpleTransformProvenance(SimpleTransform host) {
            this.op = new EnumProvenance<>(OP,host.op);
            this.operand = new DoubleProvenance(OPERAND,host.operand);
            this.secondOperand = new DoubleProvenance(SECOND_OPERAND,host.secondOperand);
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        @SuppressWarnings("unchecked") // Enum cast
        public SimpleTransformProvenance(Map<String,Provenance> map) {
            op = ObjectProvenance.checkAndExtractProvenance(map,OP,EnumProvenance.class, SimpleTransformProvenance.class.getSimpleName());
            operand = ObjectProvenance.checkAndExtractProvenance(map,OPERAND,DoubleProvenance.class, SimpleTransformProvenance.class.getSimpleName());
            secondOperand = ObjectProvenance.checkAndExtractProvenance(map,SECOND_OPERAND,DoubleProvenance.class,SimpleTransformProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return SimpleTransform.class.getName();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof SimpleTransformProvenance)) return false;
            SimpleTransformProvenance pairs = (SimpleTransformProvenance) o;
            return op.equals(pairs.op) &&
                    operand.equals(pairs.operand) &&
                    secondOperand.equals(pairs.secondOperand);
        }

        @Override
        public int hashCode() {
            return Objects.hash(op, operand, secondOperand);
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            Map<String,Provenance> map = new HashMap<>();
            map.put(OP,op);
            map.put(OPERAND,operand);
            map.put(SECOND_OPERAND,secondOperand);
            return Collections.unmodifiableMap(map);
        }
    }

    /**
     * No-op on this TransformStatistics.
     * @param value The value to observe
     */
    @Override
    public void observeValue(double value) { }

    /**
     * No-op on this TransformStatistics.
     */
    @Override
    @Deprecated
    public void observeSparse() { }

    /**
     * No-op on this TransformStatistics.
     */
    @Override
    public void observeSparse(int count) { }

    /**
     * Returns itself.
     * @return this.
     */
    @Override
    public Transformer generateTransformer() {
        return this;
    }

    /**
     * Returns itself.
     * @return this.
     */
    @Override
    public TransformStatistics createStats() {
        return this;
    }

    /**
     * Apply the operation to the input.
     * @param input The input value to transform.
     * @return The transformed value.
     */
    @Override
    public double transform(double input) {
        return operation.applyAsDouble(input);
    }

    @Override
    public String toString() {
        switch (op) {
            case exp:
                return "exp()";
            case log:
                return "log()";
            case add:
                return "add("+operand+")";
            case sub:
                return "sub("+operand+")";
            case mul:
                return "mul("+operand+")";
            case div:
                return "div("+operand+")";
            case binarise:
                return "binarise()";
            case threshold:
                return "threshold(min="+operand+",max="+secondOperand+")";
            default:
                return op.toString();
        }
    }

    /**
     * Generate a SimpleTransform that applies
     * {@link Math#exp}.
     * @return The exponential function.
     */
    public static SimpleTransform exp() {
        return new SimpleTransform(Operation.exp);
    }

    /**
     * Generate a SimpleTransform that applies
     * {@link Math#log}.
     * @return The logarithm function.
     */
    public static SimpleTransform log() {
        return new SimpleTransform(Operation.log);
    }

    /**
     * Generate a SimpleTransform that
     * adds the operand to each value.
     * @param operand The operand to add.
     * @return An addition function.
     */
    public static SimpleTransform add(double operand) {
        return new SimpleTransform(Operation.add,operand);
    }

    /**
     * Generate a SimpleTransform that
     * subtracts the operand from each value.
     * @param operand The operand to subtract.
     * @return A subtraction function.
     */
    public static SimpleTransform sub(double operand) {
        return new SimpleTransform(Operation.sub,operand);
    }

    /**
     * Generate a SimpleTransform that
     * multiplies each value by the operand.
     * @param operand The operand to multiply.
     * @return A multiplication function.
     */
    public static SimpleTransform mul(double operand) {
        return new SimpleTransform(Operation.mul,operand);
    }

    /**
     * Generate a SimpleTransform that
     * divides each value by the operand.
     * @param operand The divisor.
     * @return A division function.
     */
    public static SimpleTransform div(double operand) {
        return new SimpleTransform(Operation.div,operand);
    }

    /**
     * Generate a SimpleTransform that sets negative and
     * zero values to zero and positive values to one.
     * @return A binarising function.
     */
    public static SimpleTransform binarise() {
        return new SimpleTransform(Operation.binarise);
    }

    /**
     * Generate a SimpleTransform that sets values below min to
     * min, and values above max to max.
     * @param min The minimum value. To not threshold below, set to {@link Double#NEGATIVE_INFINITY}.
     * @param max The maximum value. To not threshold above, set to {@link Double#POSITIVE_INFINITY}.
     * @return A thresholding function.
     */
    public static SimpleTransform threshold(double min, double max) {
        return new SimpleTransform(Operation.threshold,min,max);
    }

    /**
     * Tag interface to make the operators serializable.
     */
    interface SerializableDoubleUnaryOperator extends DoubleUnaryOperator, Serializable {}
}

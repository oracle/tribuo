package org.tribuo.classification.fs.wrapper.Discreeting;

import java.util.function.DoubleUnaryOperator;

import static org.apache.commons.math3.special.Erf.erf;

/**
 * Enumeration that contains the types of transfer functions in which they are used to define the type of transfer function
 */
public enum TransferFunction implements DoubleUnaryOperator {
        V1, V2, V3, V4, S1, S2, S3, S4;

        /**
         * Applies this operator to the given value.
         *
         * @param value the operand as continuous value to be converted to either 1 or 0
         * @return the operator result that is a d
         */
        @Override
        public double applyAsDouble(double value) {
                return switch (this) {
                        case V1 -> Math.abs(erf(Math.sqrt(Math.PI) / 2 * value)) >= 0.5 ? 1 : 0;
                        case V2 -> Math.abs(Math.tan(value)) >= 0.5 ? 1 : 0;
                        case V3 -> Math.abs(value / Math.abs(1 + Math.pow(value, 2))) >= 0.5 ? 1 : 0;
                        case V4 -> Math.abs(2 / Math.PI * Math.atan(Math.PI / 2 * value)) >= 0.5 ? 1 : 0;
                        case S1 -> 1 / (1 + Math.pow(Math.E, - 2 * value)) >= 0.5 ? 1 : 0;
                        case S2 -> 1 / (1 + Math.pow(Math.E, - value)) >= 0.5 ? 1 : 0;
                        case S3 -> 1 / (1 + Math.pow(Math.E, - value / 2)) >= 0.5 ? 1 : 0;
                        case S4 -> 1 / (1 + Math.pow(Math.E, - value / 3)) >= 0.5 ? 1 : 0;
                };
        }
}


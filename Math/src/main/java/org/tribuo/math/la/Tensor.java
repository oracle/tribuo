/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.la;

import org.tribuo.math.protos.TensorProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;

import java.io.Serializable;
import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * An interface for Tensors, currently Vectors and Matrices.
 */
public interface Tensor extends ProtoSerializable<TensorProto>, Serializable {

    /**
     * The number of elements in this shape, i.e., the product of the shape array.
     * @param shape The tensor shape.
     * @return The total number of elements.
     */
    public static int shapeSum(int[] shape) {
        int sum = 1;
        for (int i = 0; i < shape.length; i++) {
            sum *= shape[i];
        }
        return sum;
    }

    /**
     * Checks that the two tensors have compatible shapes.
     * <p>
     * Compatible shapes are those which are exactly equal, as Tribuo does not
     * support broadcasting.
     * @param first The first tensor.
     * @param second The second tensor.
     * @return True if the shapes are the same.
     */
    public static boolean shapeCheck(Tensor first, Tensor second) {
        if ((first != null) && (second != null)) {
            return Arrays.equals(first.getShape(),second.getShape());
        } else {
            return false;
        }
    }

    /**
     * Returns an int array specifying the shape of this {@link Tensor}.
     * @return An int array.
     */
    public int[] getShape();

    /**
     * Reshapes the Tensor to the supplied shape. Throws {@link IllegalArgumentException} if the shape isn't compatible.
     * @param shape The desired shape.
     * @return A Tensor of the desired shape.
     */
    public Tensor reshape(int[] shape);

    /**
     * Returns a copy of this Tensor.
     * @return A copy of the Tensor.
     */
    public Tensor copy();

    /**
     * Updates this {@link Tensor} by adding all the values from the intersection with {@code other}.
     * <p>
     * The function {@code f} is applied to all values from {@code other} before the
     * addition.
     * <p>
     * Each value is updated as value += f(otherValue).
     * @param other The other {@link Tensor}.
     * @param f A function to apply.
     */
    public void intersectAndAddInPlace(Tensor other, DoubleUnaryOperator f);

    /**
     * Same as {@link Tensor#intersectAndAddInPlace}, but applies the identity function.
     * <p>
     * Each value is updated as value += otherValue.
     * @param other The other {@link Tensor}.
     */
    default public void intersectAndAddInPlace(Tensor other) {
        intersectAndAddInPlace(other, DoubleUnaryOperator.identity());
    }

    /**
     * Updates this {@link Tensor} with the Hadamard product
     * (i.e., a term by term multiply) of this and {@code other}.
     * <p>
     * The function {@code f} is applied to all values from {@code other} before the addition.
     * <p>
     * Each value is updated as value *= f(otherValue).
     * @param other The other {@link Tensor}.
     * @param f A function to apply.
     */
    public void hadamardProductInPlace(Tensor other, DoubleUnaryOperator f);

    /**
     * Same as {@link Tensor#hadamardProductInPlace}, but applies the identity function.
     * <p>
     * Each value is updated as value *= otherValue.
     * @param other The other {@link Tensor}.
     */
    default public void hadamardProductInPlace(Tensor other) {
        hadamardProductInPlace(other, DoubleUnaryOperator.identity());
    }

    /**
     * Applies a {@link DoubleUnaryOperator} elementwise to this {@link Tensor}.
     * @param f The function to apply.
     */
    public void foreachInPlace(DoubleUnaryOperator f);

    /**
     * Scales each element of this {@link Tensor} by {@code coefficient}.
     * @param coefficient The coefficient of scaling.
     */
    default public void scaleInPlace(double coefficient) {
        foreachInPlace(d -> d * coefficient);
    }

    /**
     * Adds {@code scalar} to each element of this {@link Tensor}.
     * @param scalar The scalar to add.
     */
    default public void scalarAddInPlace(double scalar) {
        foreachInPlace(d -> d + scalar);
    }

    /**
     * Calculates the euclidean norm for this vector.
     * @return The euclidean norm.
     */
    public double twoNorm();

    /**
     * Deserialize a tensor proto into a Tensor.
     * <p>
     * Throws {@link IllegalArgumentException} if the proto is invalid.
     * @param proto The proto to deserialize.
     * @return The tensor.
     */
    public static Tensor deserialize(TensorProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}

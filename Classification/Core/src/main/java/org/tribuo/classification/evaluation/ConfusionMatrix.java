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

package org.tribuo.classification.evaluation;

import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Classifiable;

import java.util.function.ToDoubleFunction;

/**
 * A confusion matrix for {@link Classifiable}s.
 *
 * <p>
 * We interpret it as follows:
 *
 * {@code
 * C[i, j] = k
 * }
 *
 * means "the TRUE class 'j' was PREDICTED to be class 'i' a total of 'k' times".
 *
 * <p>
 * In other words, the row indices correspond to the model's predictions, and the column indices correspond to
 * the ground truth.
 * </p>
 * @param <T> The type of the output.
 */
public interface ConfusionMatrix<T extends Classifiable<T>> {

    /**
     * Returns the classification domain that this confusion matrix operates over.
     * @return The classification domain.
     */
    public ImmutableOutputInfo<T> getDomain();

    /**
     * The number of examples this confusion matrix has seen.
     * @return The number of examples.
     */
    public double support();

    /**
     * The number of examples with this true label this confusion matrix has seen.
     * @param cls The label.
     * @return The number of examples.
     */
    public double support(T cls);

    /**
     * The number of true positives for the supplied label.
     * @param cls The label.
     * @return The number of examples.
     */
    public double tp(T cls);

    /**
     * The number of false positives for the supplied label.
     * @param cls The label.
     * @return The number of examples.
     */
    public double fp(T cls);

    /**
     * The number of false negatives for the supplied label.
     * @param cls The label.
     * @return The number of examples.
     */
    public double fn(T cls);

    /**
     * The number of true negatives for the supplied label.
     * @param cls The label.
     * @return The number of examples.
     */
    public double tn(T cls);

    /**
     * The number of times the supplied predicted label was returned for the supplied true class.
     * @param predictedLabel The predicted label.
     * @param trueLabel The true label.
     * @return The number of examples predicted as {@code predictedLabel} when the true label was {@code trueLabel}.
     */
    public double confusion(T predictedLabel, T trueLabel);

    /**
     * The total number of true positives.
     * @return The total true positives.
     */
    public default double tp() {
        return sumOverOutputs(getDomain(), this::tp);
    }

    /**
     * The total number of false positives.
     * @return The total false positives.
     */
    public default double fp() {
        return sumOverOutputs(getDomain(), this::fp);
    }

    /**
     * The total number of false negatives.
     * @return The total false negatives.
     */
    public default double fn() {
        return sumOverOutputs(getDomain(), this::fn);
    }

    /**
     * The total number of true negatives.
     * @return The total true negatives.
     */
    public default double tn() {
        return sumOverOutputs(getDomain(), this::tn);
    }

    /**
     * Sums the supplied getter over the domain.
     * @param domain The domain to sum over.
     * @param getter The getter to use.
     * @param <T> The type of the output.
     * @return The total summed over the domain.
     */
    static <T extends Classifiable<T>> double sumOverOutputs(ImmutableOutputInfo<T> domain, ToDoubleFunction<T> getter) {
        double total = 0;
        for (T key : domain.getDomain()) {
            total += getter.applyAsDouble(key);
        }
        return total;
    }
}

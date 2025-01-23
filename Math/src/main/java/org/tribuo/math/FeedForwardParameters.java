/*
 * Copyright (c) 2021, 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math;

import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;

/**
 * A Parameters for models which make a single prediction like logistic regressions and neural networks.
 */
public interface FeedForwardParameters extends Parameters {

    /**
     * Generates an un-normalized prediction by feeding the features through the parameters.
     * @param example The features.
     * @return The prediction.
     */
    public DenseVector predict(SGDVector example);

    /**
     * Generates a batch of un-normalized predictions by feeding the features through the parameters.
     * @param batch The feature matrix.
     * @return The prediction.
     */
    public DenseMatrix predict(Matrix batch);

    /**
     * Generates the parameter gradients given the loss, output gradient and input
     * features.
     * @param score The loss and gradient.
     * @param features The input features.
     * @return The parameter gradient array.
     */
    public Tensor[] gradients(LossAndGrad score, SGDVector features);

    /**
     * Generates the parameter gradients given the loss, output gradient and input
     * feature batch.
     * @param score The loss and gradient.
     * @param batch The input features.
     * @return The parameter gradient array.
     */
    public Tensor[] gradients(BatchLossAndGrad score, Matrix batch);

    /**
     * Returns a copy of the parameters.
     * @return A copy of the model parameters.
     */
    public FeedForwardParameters copy();

}

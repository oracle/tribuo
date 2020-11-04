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

package org.tribuo.regression.evaluation;

import org.tribuo.evaluation.Evaluation;
import org.tribuo.regression.Regressor;

import java.util.Map;

/**
 * Defines methods that calculate regression performance.
 */
public interface RegressionEvaluation extends Evaluation<Regressor> {

    /**
     * The average Mean Absolute Error across all dimensions.
     * @return The average Mean Absolute Error.
     */
    double averageMAE();

    /**
     * Calculates the Mean Absolute Error for that dimension.
     * @param variable The regression dimension to use.
     * @return The Mean Absolute Error.
     */
    double mae(Regressor variable);

    /**
     * Calculates the Mean Absolute Error for all dimensions.
     * @return The Mean Absolute Error.
     */
    Map<Regressor, Double> mae();

    /**
     * The average R2 across all dimensions.
     * @return The average R2.
     */
    double averageR2();

    /**
     * Calculates R2 for the supplied dimension.
     * @param variable The regression dimension to use.
     * @return The R2.
     */
    double r2(Regressor variable);

    /**
     * Calculates R2 for all dimensions.
     * @return The R2.
     */
    Map<Regressor, Double> r2();

    /**
     * The average RMSE across all dimensions.
     * @return The average RMSE.
     */
    double averageRMSE();

    /**
     * Calculates the Root Mean Squared Error (i.e., the square root of the average squared errors across all data points) for the supplied dimension.
     * @param variable The regression dimension to use.
     * @return The RMSE.
     */
    double rmse(Regressor variable);

    /**
     * Calculates the RMSE for all dimensions.
     * @return The RMSE.
     */
    Map<Regressor, Double> rmse();

    /**
     * The average explained variance across all dimensions.
     * @return The average explained variance.
     */
    double averagedExplainedVariance();

    /**
     * Calculates the explained variance of the ground truth using the predictions for the supplied dimension.
     * @param variable The regression dimension to use.
     * @return The explained variance.
     */
    double explainedVariance(Regressor variable);

    /**
     * Calculatest the explained variance for all dimensions.
     * @return The explained variance.
     */
    Map<Regressor, Double> explainedVariance();

}
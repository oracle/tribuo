/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.liblinear;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Output;
import de.bwaldvogel.liblinear.SolverType;

import java.io.Serializable;

/**
 * A carrier type for the liblinear algorithm type. It really wants to be a set of enums with
 * different type parameters, but it's encoded as an interface where each
 * subclass for an {@link Output} implementation contains an enum with it's
 * valid values.
 * <p>
 * LibLinear supported enum values for various tasks are:
 * <ul>
 * <li>L2R_LR - L2-regularized logistic regression (primal)</li>
 * <li>L2R_L2LOSS_SVC_DUAL - L2-regularized L2-loss support vector classification (dual)</li>
 * <li>L2R_L2LOSS_SVC - L2-regularized L2-loss support vector classification (primal)</li>
 * <li>L2R_L1LOSS_SVC_DUAL - L2-regularized L1-loss support vector classification (dual)</li>
 * <li>MCSVM_CS - multi-class support vector classification by Crammer and Singer</li>
 * <li>L1R_L2LOSS_SVC - L1-regularized L2-loss support vector classification</li>
 * <li>L1R_LR - L1-regularized logistic regression</li>
 * <li>L2R_LR_DUAL - L2-regularized logistic regression (dual)</li>
 * <li>L2R_L2LOSS_SVR - L2-regularized L2-loss support vector regression (primal)</li>
 * <li>L2R_L2LOSS_SVR_DUAL - L2-regularized L2-loss support vector regression (dual)</li>
 * <li>L2R_L1LOSS_SVR_DUAL - L2-regularized L1-loss support vector regression (dual)</li>
 * </ul>
 */
public interface LibLinearType<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {
    /*
    // L2-regularized logistic regression (primal)
    L2R_LR,
    // L2-regularized L2-loss support vector classification (dual)
    L2R_L2LOSS_SVC_DUAL,
    // L2-regularized L2-loss support vector classification (primal)
    L2R_L2LOSS_SVC,
    // L2-regularized L1-loss support vector classification (dual)
    L2R_L1LOSS_SVC_DUAL,
    // multi-class support vector classification by Crammer and Singer
    MCSVM_CS,
    // L1-regularized L2-loss support vector classification
    L1R_L2LOSS_SVC,
    // L1-regularized logistic regression
    L1R_LR,
    // L2-regularized logistic regression (dual)
    L2R_LR_DUAL,
    // L2-regularized L2-loss support vector regression (primal)
    L2R_L2LOSS_SVR,
    // L2-regularized L2-loss support vector regression (dual)
    L2R_L2LOSS_SVR_DUAL,
    // L2-regularized L1-loss support vector regression (dual)
    L2R_L1LOSS_SVR_DUAL,
    // One-class SVM
    ONECLASS_SVM;
    */

    /**
     * Is this class a Classification algorithm?
     * @return True if it's a classification algorithm.
     */
    public boolean isClassification();

    /**
     * Is this class a Regression algorithm?
     * @return True if it's a regression algorithm.
     */
    public boolean isRegression();

    /**
     * Is this class an anomaly detection algorithm?
     * @return True if it's an anomaly detection algorithm.
     */
    public boolean isAnomaly();

    /**
     * Returns the liblinear enum type.
     * @return The liblinear enum type.
     */
    public SolverType getSolverType();
}

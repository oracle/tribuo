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

package org.tribuo.regression.liblinear;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.common.liblinear.LibLinearType;
import org.tribuo.regression.Regressor;
import de.bwaldvogel.liblinear.SolverType;

import java.io.Serializable;

/**
 * The carrier type for liblinear linear regression modes.
 * <p>
 * Supports: L2R_L2LOSS_SVR, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL.
 */
public final class LinearRegressionType implements LibLinearType<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * The type of linear regression algorithm.
     */
    public enum LinearType implements Serializable {
        /**
         * L2-regularized L2-loss support vector regression (primal)
         */
        L2R_L2LOSS_SVR(SolverType.L2R_L2LOSS_SVR),
        /**
         * L2-regularized L2-loss support vector regression (dual)
         */
        L2R_L2LOSS_SVR_DUAL(SolverType.L2R_L2LOSS_SVR_DUAL),
        /**
         * L2-regularized L1-loss support vector regression (dual)
         */
        L2R_L1LOSS_SVR_DUAL(SolverType.L2R_L1LOSS_SVR_DUAL);

        private final SolverType type;

        LinearType(SolverType type) {
            this.type = type;
        }

        /**
         * Returns the liblinear enum.
         * @return The liblinear enum.
         */
        public SolverType getSolverType() {
            return type;
        }
    }

    @Config(description="The type of regression algorithm.",mandatory = true)
    private LinearType type;

    /**
     * For olcut.
     */
    private LinearRegressionType() {}

    /**
     * Constructs a LinearRegressionType with the specified LibLinear algorithm.
     * @param type The liblinear algorithm.
     */
    public LinearRegressionType(LinearType type) {
        this.type = type;
    }

    @Override
    public boolean isClassification() {
        return false;
    }

    @Override
    public boolean isRegression() {
        return true;
    }

    @Override
    public boolean isAnomaly() {
        return false;
    }

    @Override
    public SolverType getSolverType() {
        return type.getSolverType();
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"LibLinearType");
    }

}

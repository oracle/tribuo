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

package org.tribuo.classification.liblinear;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.classification.Label;
import org.tribuo.common.liblinear.LibLinearType;
import de.bwaldvogel.liblinear.SolverType;

import java.io.Serializable;

/**
 * The carrier type for liblinear classification modes.
 * <p>
 * Supports: L1R_L2LOSS_SVC, L2R_L2LOSS_SVC, L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_LR, L2R_LR, L2R_LR_DUAL.
 */
public final class LinearClassificationType implements LibLinearType<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * The different model types available for classification.
     */
    public enum LinearType implements Serializable {
        /**
         * L1-regularized L2-loss support vector classification
         */
        L1R_L2LOSS_SVC(SolverType.L1R_L2LOSS_SVC),
        /**
         * L2-regularized L2-loss support vector classification (primal)
         */
        L2R_L2LOSS_SVC(SolverType.L2R_L2LOSS_SVC),
        /**
         * L2-regularized L2-loss support vector classification (dual)
         */
        L2R_L2LOSS_SVC_DUAL(SolverType.L2R_L2LOSS_SVC_DUAL),
        /**
         * L2-regularized L1-loss support vector classification (dual)
         */
        L2R_L1LOSS_SVC_DUAL(SolverType.L2R_L1LOSS_SVC_DUAL),
        /**
         * multi-class support vector classification by Crammer and Singer
         */
        MCSVM_CS(SolverType.MCSVM_CS),
        /**
         * L1-regularized logistic regression
         */
        L1R_LR(SolverType.L1R_LR),
        /**
         * L2-regularized logistic regression (primal)
         */
        L2R_LR(SolverType.L2R_LR),
        /**
         * L2-regularized logistic regression (dual)
         */
        L2R_LR_DUAL(SolverType.L2R_LR_DUAL);

        private final SolverType type;

        LinearType(SolverType type) {
            this.type = type;
        }

        /**
         * Gets the LibLinear solver type.
         * @return The solver type.
         */
        public SolverType getSolverType() {
            return type;
        }
    }

    @Config(mandatory=true, description = "The type of classification model")
    private LinearType type;

    /**
     * For olcut.
     */
    private LinearClassificationType() {}

    /**
     * Constructs a LinearClassificationType using the supplied algorithm.
     * @param type The liblinear algorithm.
     */
    public LinearClassificationType(LinearType type) {
        this.type = type;
    }

    @Override
    public boolean isClassification() {
        return true;
    }

    @Override
    public boolean isRegression() {
        return false;
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

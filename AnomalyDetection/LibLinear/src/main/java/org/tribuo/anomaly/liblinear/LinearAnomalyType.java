/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.anomaly.liblinear;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.anomaly.Event;
import org.tribuo.common.liblinear.LibLinearType;
import de.bwaldvogel.liblinear.SolverType;

import java.io.Serializable;

/**
 * The carrier type for liblinear anomaly detection modes.
 * <p>
 * Supports: ONECLASS_SVM
 */
public final class LinearAnomalyType implements LibLinearType<Event> {
    private static final long serialVersionUID = 1L;

    /**
     * The different model types available for classification.
     */
    public enum LinearType implements Serializable {
        /**
         * Linear one-class SVM
         */
        ONECLASS_SVM(SolverType.ONECLASS_SVM);

        private final SolverType type;

        LinearType(SolverType type) {
            this.type = type;
        }

        /**
         * Gets the type of the solver.
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
    private LinearAnomalyType() {}

    /**
     * Constructs the type of the liblinear anomaly detector.
     * @param type The anomaly detector type.
     */
    public LinearAnomalyType(LinearType type) {
        this.type = type;
    }

    @Override
    public boolean isClassification() {
        return false;
    }

    @Override
    public boolean isRegression() {
        return false;
    }

    @Override
    public boolean isAnomaly() {
        return true;
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

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

package org.tribuo.regression.libsvm;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.common.libsvm.SVMType;
import org.tribuo.regression.Regressor;

/**
 * The carrier type for LibSVM regression modes.
 * <p>
 * Supports C_SVC and NU_SVC.
 */
public class SVMRegressionType implements SVMType<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Type of regression SVM.
     */
    public enum SVMMode {
        /**
         * epsilon-insensitive SVR.
         */
        EPSILON_SVR(3),
        /**
         * optimization in dual space.
         */
        NU_SVR(4);

        final int nativeType;

        SVMMode(int type) {
            this.nativeType = type;
        }
    }

    @Config(mandatory=true,description="The SVM regression algorithm to use.")
    private SVMMode type;

    /**
     * for olcut.
     */
    private SVMRegressionType() {}

    /**
     * Constructs an SVMRegressionType using the specified SVM algorithm.
     * @param type The SVM algorithm.
     */
    public SVMRegressionType(SVMMode type) {
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
    public boolean isNu() {
        return type == SVMMode.NU_SVR;
    }

    @Override
    public int getNativeType() {
        return type.nativeType;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"SVMType");
    }
}

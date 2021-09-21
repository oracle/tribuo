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

package org.tribuo.classification.libsvm;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.classification.Label;
import org.tribuo.common.libsvm.SVMType;

/**
 * The carrier type for LibSVM classification modes.
 * <p>
 * Supports C_SVC and NU_SVC.
 */
public class SVMClassificationType implements SVMType<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * The classification model types.
     */
    public enum SVMMode {
        /**
         * Original SVM algorithm.
         */
        C_SVC(0),
        /**
         * Classification SVM, optimization in dual space.
         */
        NU_SVC(1);

        final int nativeType;

        SVMMode(int type) {
            this.nativeType = type;
        }
    }

    @Config(mandatory=true,description="The SVM classification algorithm to use.")
    private SVMMode type;

    /**
     * for olcut.
     */
    private SVMClassificationType() {}

    /**
     * Constructs an SVMClassificationType using the supplied SVM algorithm.
     * @param type The SVM algorithm.
     */
    public SVMClassificationType(SVMMode type) {
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
    public boolean isNu() {
        return type == SVMMode.NU_SVC;
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

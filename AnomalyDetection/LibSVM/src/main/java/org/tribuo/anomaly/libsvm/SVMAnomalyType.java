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

package org.tribuo.anomaly.libsvm;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.anomaly.Event;
import org.tribuo.common.libsvm.SVMType;

/**
 * The carrier type for LibSVM anomaly detection modes.
 * <p>
 * Supports ONE_CLASS. Yes it's a single value enum.
 */
public class SVMAnomalyType implements SVMType<Event> {
    private static final long serialVersionUID = 1L;

    /**
     * Valid SVM modes for anomaly detection.
     */
    public enum SVMMode {
        /**
         * Anomaly detection SVM.
         */
        ONE_CLASS(2);

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
    private SVMAnomalyType() {}

    /**
     * Constructs an SVM anomaly type wrapping the SVM algorithm choice.
     * @param type The svm algorithm type.
     */
    public SVMAnomalyType(SVMMode type) {
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
    public boolean isNu() {
        return true;
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

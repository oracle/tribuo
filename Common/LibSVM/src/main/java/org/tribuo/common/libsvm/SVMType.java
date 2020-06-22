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

package org.tribuo.common.libsvm;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Output;

import java.io.Serializable;

/**
 * A carrier type for the SVM type. It really wants to be a set of enums with
 * different type parameters, but it's encoded as an interface where each
 * subclass for an {@link Output} implementation contains an enum with it's
 * valid values.
 * <p>
 * LibSVM supported enum values are:
 * <ul>
 * <li>C_SVC(0) - Original SVM algorithm.</li>
 * <li>NU_SVC(1) - Original SVM, optimization in dual space.</li>
 * <li>ONE_CLASS(2) - Anomaly detection SVM.</li>
 * <li>EPSILON_SVR(3) - epsilon-insensitive SVR.</li>
 * <li>NU_SVR(4) - nu-SVR, optimization in dual space.</li>
 * </ul>
 */
public interface SVMType<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {
    /**
     * Is this a classification algorithm.
     * @return True if it's a classification algorithm.
     */
    public boolean isClassification();

    /**
     * Is this a regression algorithm.
     * @return True if it's a regression algorithm.
     */
    public boolean isRegression();

    /**
     * Is this an anomaly detection algorithm.
     * @return True if it's an anomaly detection algorithm.
     */
    public boolean isAnomaly();

    /**
     * Is this a nu-SVM.
     * @return True if it's a nu-SVM.
     */
    public boolean isNu();

    /**
     * The LibSVM int id for the algorithm.
     * @return The int id.
     */
    public int getNativeType();
}

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

package org.tribuo.math.kernel;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.la.SparseVector;

/**
 * A polynomial kernel, (gamma*u.dot(v) + intercept)^degree.
 */
public class Polynomial implements Kernel {
    private static final long serialVersionUID = 1L;

    @Config(mandatory = true,description="Coefficient to multiply the dot product by.")
    private double gamma;

    @Config(mandatory = true,description="Scalar to add to the dot product.")
    private double intercept;

    @Config(mandatory = true,description="Degree of the polynomial.")
    private double degree;

    /**
     * For olcut.
     */
    private Polynomial() {}

    /**
     * A polynomial kernel, (gamma*u.dot(v) + intercept)^degree.
     * @param gamma The scalar coefficient.
     * @param intercept An additive coefficient.
     * @param degree The degree of the polynomial.
     */
    public Polynomial(double gamma, double intercept, double degree) {
        this.gamma = gamma;
        this.intercept = intercept;
        this.degree = degree;
    }

    @Override
    public double similarity(SparseVector a, SparseVector b) {
        return Math.pow(gamma * a.dot(b) + intercept, degree);
    }

    @Override
    public String toString() {
        return "Polynomial(gamma="+gamma+",intercept="+intercept+",degree="+degree+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"Kernel");
    }
}

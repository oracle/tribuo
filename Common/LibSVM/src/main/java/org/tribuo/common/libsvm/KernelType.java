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

import java.io.Serializable;

/**
 * Kernel types from libsvm.
 */
public enum KernelType implements Serializable {
    /**
     * A linear kernel function (i.e., a dot product).
     */
    LINEAR(0),
    /**
     * A polynomial kernel of the form (gamma*u'*v + coef0)^degree
     */
    POLY(1),
    /**
     * An RBF kernel of the form exp(-gamma*|u-v|^2)
     */
    RBF(2),
    /**
     * A sigmoid kernel of the form tanh(gamma*u'*v + coef0)
     */
    SIGMOID(3);

    final int nativeType;

    KernelType(int nativeType) {
        this.nativeType = nativeType;
    }

    /**
     * Gets LibSVM's int id.
     * @return The int id.
     */
    public int getNativeType() {
        return nativeType;
    }

    /**
     * Converts the LibSVM int id into the enum value.
     * @param nativeType The LibSVM id.
     * @return The corresponding enum.
     */
    public static KernelType getKernelType(int nativeType) {
        switch (nativeType) {
            case 0:
                return LINEAR;
            case 1:
                return POLY;
            case 2:
                return RBF;
            case 3:
                return SIGMOID;
            default:
                throw new IllegalArgumentException("Unknown native type " + nativeType);
        }
    }
}

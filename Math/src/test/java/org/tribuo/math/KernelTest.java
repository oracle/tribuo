/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math;

import org.junit.jupiter.api.Test;
import org.tribuo.math.kernel.Linear;
import org.tribuo.math.kernel.Polynomial;
import org.tribuo.math.kernel.RBF;
import org.tribuo.math.kernel.Sigmoid;

import static org.tribuo.test.Helpers.testProtoSerialization;

public class KernelTest {

    @Test
    public void testLinear() {
        Linear lin = new Linear();
        testProtoSerialization(lin);
    }

    @Test
    public void testRBF() {
        RBF lin = new RBF(0.5);
        testProtoSerialization(lin);
    }

    @Test
    public void testPolynomial() {
        Polynomial lin = new Polynomial(0.5,1,3);
        testProtoSerialization(lin);
    }

    @Test
    public void testSigmoid() {
        Sigmoid lin = new Sigmoid(0.25,3);
        testProtoSerialization(lin);
    }

}

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

package org.tribuo.math.optimisers.util;

import org.junit.jupiter.api.Test;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseMatrixTest;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.DenseVectorTest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.test.Helpers.testProtoSerialization;

public class ShrinkingTensorTest {

    @Test
    public void matrixSerializationTest() {
        DenseMatrix a = DenseMatrixTest.generateA();
        ShrinkingMatrix sh = new ShrinkingMatrix(a,0.1,true);
        ShrinkingMatrix deser = testProtoSerialization(sh);
        assertEquals(deser.getClass(),sh.getClass());
    }

    @Test
    public void vectorSerializationTest() {
        DenseVector a = DenseVectorTest.generateVectorA();
        ShrinkingVector sh = new ShrinkingVector(a,0.1,true);
        ShrinkingVector deser = testProtoSerialization(sh);
        assertEquals(deser.getClass(),sh.getClass());
    }

}

/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.embeddings;

import java.nio.FloatBuffer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 *
 */
public class FloatTensorBufferTest {

    @Test
    public void l2Test() {
        float[] input = new float[]{0,1,2,3,4,5,6,7,8,9,10,11};
        FloatBuffer buf = FloatBuffer.allocate(input.length);
        buf.put(input);
        FloatTensorBuffer tens = new FloatTensorBuffer(buf, new long[]{2,2,3});
        tens.l2InPlace();
        float[] firstRow = new float[]{tens.get(0,0,0), tens.get(0,0,1), tens.get(0,0,2)};
        Assertions.assertEquals(1.0, vecLength(firstRow), 1e-5);
        float[] secondRow = new float[]{tens.get(0,1,0), tens.get(0,1,1), tens.get(0,1,2)};
        Assertions.assertEquals(1.0, vecLength(secondRow), 1e-5);
        float[] thirdRow = new float[]{tens.get(1,0,0), tens.get(1,0,1), tens.get(1,0,2)};
        Assertions.assertEquals(1.0, vecLength(thirdRow), 1e-5);
        float[] fourthRow = new float[]{tens.get(1,1,0), tens.get(1,1,1), tens.get(1,1,2)};
        Assertions.assertEquals(1.0, vecLength(fourthRow), 1e-5);
    }

    private static float vecLength(float[] input) {
        float total = 0.0f;
        for (int i = 0; i < input.length; i++) {
            total += input[i] * input[i];
        }
        return (float) Math.sqrt(total);
    }

}

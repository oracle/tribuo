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

package org.tribuo.util.infotheory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import static org.tribuo.util.infotheory.Gamma.gamma;

public class GammaTest {

    @Test
    public void testExamples() {
        assertEquals(Double.NaN, gamma(0.0));
        assertEquals(1.77245385, gamma(0.5), 1e-8);
        assertEquals(1.0, gamma(1.0));
        assertEquals(24.0, gamma(5.0), 1e-8);

        //some random examples betwixt -100 and 100
        assertEquals(8.06474995572965e+79, gamma(59.86728989339031), 1e+67);
        assertEquals(0.0005019871198070064, gamma(-7.260823951121694), 1e-18);
        assertEquals(1.5401131084717308e-110, gamma(-75.48705446197417), 1e-124);
        assertEquals(95932082427.69138, gamma(15.035762406520718), 1e-3);
        assertEquals(4.2868413548339677e+154, gamma(99.32984689647557), 1e+140);
        assertEquals(-4.971777508910858e-48, gamma(-40.14784332381653), 1e-60);
        assertEquals(5.3603547985340755e-96, gamma(-67.85881128534656), 1e-108);
        assertEquals(-1.887428186224555e-151, gamma(-96.63801919072759), 1e-163);
        assertEquals(6.0472720813564265e+125, gamma(84.61636884564746), 1e+113);
        assertEquals(-7.495823228458869e-128, gamma(-84.57833815656579), 1e-140);
        assertEquals(-2.834337137147687e-14, gamma(-16.831988025996992), 1e-26);
        assertEquals(8.990293245462624e+78, gamma(59.32945503543496), 1e+66);
        assertEquals(3.604695169965482e-83, gamma(-61.045472852581774), 1e-95);
        assertEquals(0.00020572694516842935, gamma(-7.545439745563854), 1e-16);
        assertEquals(-7.906506608405116e-105, gamma(-72.4403778408159), 1e-117);
        assertEquals(780133888.913568, gamma(13.192513244283958), 1e-4);
        assertEquals(-3.0601588660760365e-130, gamma(-86.09108451479372), 1e-142);
        assertEquals(2.310606358803366e+90, gamma(65.69557419730668), 1e+78);
        assertEquals(4.574728496203664e+16, gamma(19.669827320262186), 1e+4);
        assertEquals(1.5276823676246256e+74, gamma(56.618507066510915), 1e+62);
        
        assertEquals(0.0, gamma(-199.55885272585897), 1e-8);
        assertEquals(Double.POSITIVE_INFINITY, gamma(404.5418705074535));
        assertEquals(Double.NaN, gamma(-2));
    }
}

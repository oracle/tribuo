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

package org.tribuo.regression;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RegressorTest {

    @Test
    public void getsCorrectSerializableForm() {
        Regressor mr = new Regressor(
                new String[]{"a", "b", "c"},
                new double[]{1d, 2d, 3d}
        );
        assertEquals("a=1.0,b=2.0,c=3.0", mr.getSerializableForm(false));
        // Should be the same for includeConfidence either way, since we ignore NaN variances
        assertEquals("a=1.0,b=2.0,c=3.0", mr.getSerializableForm(true));

        Regressor scored = new Regressor(
                new String[]{"a", "b", "c"},
                new double[]{1d, 2d, 3d},
                new double[]{0d, 0d, 0.5}
        );
        assertEquals("a=1.0,b=2.0,c=3.0", scored.getSerializableForm(false));
        assertEquals("a=1.0\u00B10.0,b=2.0\u00B10.0,c=3.0\u00B10.5", scored.getSerializableForm(true));
    }

}
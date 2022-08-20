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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

public class InformationTheoryTest {

    /*
     * import numpy as np
     * from sklearn.metrics import mutual_info_score
     * a = np.random.randint(0,5,100)
     * #print(printArrayAsJavaDoubles(a))
     * b = np.random.randint(0,5,100)
     * #print(printArrayAsJavaDoubles(b))
     * mi = mutual_info_score(a, b)
     * print(f"mi.ln={mi}")
     * mi /= np.log(2.0)
     * print(f"mi.log2={mi}")
     */ 
    @Test
    public void testMi() {
        List<Integer> a = Arrays.asList(0, 3, 2, 3, 4, 4, 4, 1, 3, 3, 4, 3, 2, 3, 2, 4, 2, 2, 1, 4, 1, 2, 0, 4, 4, 4, 3, 3, 2, 2, 0, 4, 0, 1, 3, 0, 4, 0, 0, 4, 0, 0, 2, 2, 2, 2, 0, 3, 0, 2, 2, 3, 1, 0, 1, 0, 3, 4, 4, 4, 0, 1, 1, 3, 3, 1, 3, 4, 0, 3, 4, 1, 0, 3, 2, 2, 2, 1, 1, 2, 3, 2, 1, 3, 0, 4, 4, 0, 4, 0, 2, 1, 4, 0, 3, 0, 1, 1, 1, 0);
        List<Integer> b = Arrays.asList(4, 2, 4, 0, 4, 4, 3, 3, 3, 2, 2, 0, 1, 3, 2, 1, 2, 0, 0, 4, 3, 3, 0, 1, 1, 1, 1, 4, 4, 4, 3, 1, 0, 0, 0, 1, 4, 1, 1, 1, 3, 3, 1, 2, 3, 0, 4, 0, 2, 3, 4, 2, 3, 2, 1, 0, 2, 4, 2, 2, 4, 1, 2, 4, 3, 1, 1, 1, 3, 0, 2, 3, 2, 0, 1, 0, 0, 4, 0, 3, 0, 0, 0, 1, 3, 2, 3, 4, 2, 4, 1, 0, 3, 3, 0, 2, 1, 0, 4, 1);
        assertEquals(0.15688780624148022, InformationTheory.mi(a,b),1e-13);
    }

    /*
     * import numpy as np
     * from scipy.stats import entropy
     * a = np.random.randint(0,5,100)
     * #print(printArrayAsJavaDoubles(a))
     * hist = np.histogram(a, bins=5, density=False)[0]
     * a_probs = hist / len(a)
     * print(f"a entropy={entropy(a_probs, base=2)}")
     */
    @Test
    void testEntropy() {
        List<Integer> a = Arrays.asList(0, 3, 2, 3, 4, 4, 4, 1, 3, 3, 4, 3, 2, 3, 2, 4, 2, 2, 1, 4, 1, 2, 0, 4, 4, 4, 3, 3, 2, 2, 0, 4, 0, 1, 3, 0, 4, 0, 0, 4, 0, 0, 2, 2, 2, 2, 0, 3, 0, 2, 2, 3, 1, 0, 1, 0, 3, 4, 4, 4, 0, 1, 1, 3, 3, 1, 3, 4, 0, 3, 4, 1, 0, 3, 2, 2, 2, 1, 1, 2, 3, 2, 1, 3, 0, 4, 4, 0, 4, 0, 2, 1, 4, 0, 3, 0, 1, 1, 1, 0);
        List<Integer> b = Arrays.asList(4, 2, 4, 0, 4, 4, 3, 3, 3, 2, 2, 0, 1, 3, 2, 1, 2, 0, 0, 4, 3, 3, 0, 1, 1, 1, 1, 4, 4, 4, 3, 1, 0, 0, 0, 1, 4, 1, 1, 1, 3, 3, 1, 2, 3, 0, 4, 0, 2, 3, 4, 2, 3, 2, 1, 0, 2, 4, 2, 2, 4, 1, 2, 4, 3, 1, 1, 1, 3, 0, 2, 3, 2, 0, 1, 0, 0, 4, 0, 3, 0, 0, 0, 1, 3, 2, 3, 4, 2, 4, 1, 0, 3, 3, 0, 2, 1, 0, 4, 1);
        assertEquals(2.3167546539234776, InformationTheory.entropy(a), 1e-13);
        assertEquals(2.316147658077609, InformationTheory.entropy(b), 1e-13);
    }
    
}

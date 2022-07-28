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

package org.tribuo.clustering.evaluation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.tribuo.clustering.evaluation.ClusteringMetrics.adjustedMI;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.tribuo.util.infotheory.InformationTheory;

public class ClusteringMetricsTest {

  /*
   * import numpy as np
   * from sklearn.metrics import adjusted_mutual_info_score
   * score = adjusted_mutual_info_score([0,0,1,1], [1,0,1,1])
   *
   * a = np.random.randint(0,2,500)
   * #see printArrayAsJavaDoubles in /tribuo-math/src/test/resources/eigendecomposition-test.py
   * print(printArrayAsJavaDoubles(a))
   * b = np.random.randint(0,2,500)
   * print(printArrayAsJavaDoubles(b))
   * score = adjusted_mutual_info_score(a, b)
   */
  @Test
  void testAdjustedMI() throws Exception {
    double logBase = InformationTheory.LOG_BASE;
    InformationTheory.LOG_BASE = InformationTheory.LOG_E;
    List<Integer> a = Arrays.asList(0, 3, 2, 3, 4, 4, 4, 1, 3, 3, 4, 3, 2, 3, 2, 4, 2, 2, 1, 4, 1,
        2, 0, 4, 4, 4, 3, 3, 2, 2, 0, 4, 0, 1, 3, 0, 4, 0, 0, 4, 0, 0, 2, 2, 2, 2, 0, 3, 0, 2, 2, 3,
        1, 0, 1, 0, 3, 4, 4, 4, 0, 1, 1, 3, 3, 1, 3, 4, 0, 3, 4, 1, 0, 3, 2, 2, 2, 1, 1, 2, 3, 2, 1,
        3, 0, 4, 4, 0, 4, 0, 2, 1, 4, 0, 3, 0, 1, 1, 1, 0);
    List<Integer> b = Arrays.asList(4, 2, 4, 0, 4, 4, 3, 3, 3, 2, 2, 0, 1, 3, 2, 1, 2, 0, 0, 4, 3,
        3, 0, 1, 1, 1, 1, 4, 4, 4, 3, 1, 0, 0, 0, 1, 4, 1, 1, 1, 3, 3, 1, 2, 3, 0, 4, 0, 2, 3, 4, 2,
        3, 2, 1, 0, 2, 4, 2, 2, 4, 1, 2, 4, 3, 1, 1, 1, 3, 0, 2, 3, 2, 0, 1, 0, 0, 4, 0, 3, 0, 0, 0,
        1, 3, 2, 3, 4, 2, 4, 1, 0, 3, 3, 0, 2, 1, 0, 4, 1);
    assertEquals(0.01454420034676734, adjustedMI(a, b), 1e-14);

    a = Arrays.asList(1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
        1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1,
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
        0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
        1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 0, 0, 1);
    b = Arrays.asList(1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
        1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0,
        1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
        0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        1, 0, 0, 1, 1, 0, 0, 1, 0);
    assertEquals(-0.0014006748276587267, adjustedMI(a, b), 1e-14);

    //used to create third example
    //Random rng = new Random();
    //a = new ArrayList<>();
    //for(int i=0; i<100; i++) {
    //  int v = rng.nextDouble()*i < 20 ? 0 : i < 50 ? 1 : 2;
    //  a.add(v);
    //}
    //System.out.println(a);
    a = Arrays.asList(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 0, 2, 0, 0, 0, 0,
        2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2);
    b = Arrays.asList(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 0, 2, 0, 2, 0, 0,
        2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0,
        2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0);
    assertEquals(0.31766625364399165, adjustedMI(a, b), 1e-14);

    assertEquals(1.0, adjustedMI(Arrays.asList(0, 0, 1, 1), Arrays.asList(0, 0, 1, 1)));
    assertEquals(1.0, adjustedMI(Arrays.asList(0, 0, 1, 1), Arrays.asList(1, 1, 0, 0)));
    assertEquals(0.0, adjustedMI(Arrays.asList(0, 0, 0, 0), Arrays.asList(1, 2, 3, 4)));
    assertEquals(0.0, adjustedMI(Arrays.asList(0, 0, 1, 1), Arrays.asList(1, 1, 1, 1)));
    assertEquals(0.0834628172282441,
        adjustedMI(Arrays.asList(0, 0, 0, 1, 0, 1, 1, 1), Arrays.asList(0, 0, 0, 0, 1, 1, 1, 1)),
        1e-15);
    assertEquals(0, adjustedMI(Arrays.asList(1, 0, 1, 1), Arrays.asList(0, 0, 1, 1)), 1e-14);

    InformationTheory.LOG_BASE = logBase;
  }
}

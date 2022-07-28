package org.tribuo.util.infotheory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

public class InformationTheoryTest {

    @Test
    public void testMi() {
        List<Integer> a = Arrays.asList(0, 3, 2, 3, 4, 4, 4, 1, 3, 3, 4, 3, 2, 3, 2, 4, 2, 2, 1, 4, 1, 2, 0, 4, 4, 4, 3, 3, 2, 2, 0, 4, 0, 1, 3, 0, 4, 0, 0, 4, 0, 0, 2, 2, 2, 2, 0, 3, 0, 2, 2, 3, 1, 0, 1, 0, 3, 4, 4, 4, 0, 1, 1, 3, 3, 1, 3, 4, 0, 3, 4, 1, 0, 3, 2, 2, 2, 1, 1, 2, 3, 2, 1, 3, 0, 4, 4, 0, 4, 0, 2, 1, 4, 0, 3, 0, 1, 1, 1, 0);
        List<Integer> b = Arrays.asList(4, 2, 4, 0, 4, 4, 3, 3, 3, 2, 2, 0, 1, 3, 2, 1, 2, 0, 0, 4, 3, 3, 0, 1, 1, 1, 1, 4, 4, 4, 3, 1, 0, 0, 0, 1, 4, 1, 1, 1, 3, 3, 1, 2, 3, 0, 4, 0, 2, 3, 4, 2, 3, 2, 1, 0, 2, 4, 2, 2, 4, 1, 2, 4, 3, 1, 1, 1, 3, 0, 2, 3, 2, 0, 1, 0, 0, 4, 0, 3, 0, 0, 0, 1, 3, 2, 3, 4, 2, 4, 1, 0, 3, 3, 0, 2, 1, 0, 4, 1);
        assertEquals(0.15688780624148022, InformationTheory.mi(a,b),1e-13);
    }

    @Test
    void testEntropy() throws Exception {
        List<Integer> a = Arrays.asList(0, 3, 2, 3, 4, 4, 4, 1, 3, 3, 4, 3, 2, 3, 2, 4, 2, 2, 1, 4, 1, 2, 0, 4, 4, 4, 3, 3, 2, 2, 0, 4, 0, 1, 3, 0, 4, 0, 0, 4, 0, 0, 2, 2, 2, 2, 0, 3, 0, 2, 2, 3, 1, 0, 1, 0, 3, 4, 4, 4, 0, 1, 1, 3, 3, 1, 3, 4, 0, 3, 4, 1, 0, 3, 2, 2, 2, 1, 1, 2, 3, 2, 1, 3, 0, 4, 4, 0, 4, 0, 2, 1, 4, 0, 3, 0, 1, 1, 1, 0);
        List<Integer> b = Arrays.asList(4, 2, 4, 0, 4, 4, 3, 3, 3, 2, 2, 0, 1, 3, 2, 1, 2, 0, 0, 4, 3, 3, 0, 1, 1, 1, 1, 4, 4, 4, 3, 1, 0, 0, 0, 1, 4, 1, 1, 1, 3, 3, 1, 2, 3, 0, 4, 0, 2, 3, 4, 2, 3, 2, 1, 0, 2, 4, 2, 2, 4, 1, 2, 4, 3, 1, 1, 1, 3, 0, 2, 3, 2, 0, 1, 0, 0, 4, 0, 3, 0, 0, 0, 1, 3, 2, 3, 4, 2, 4, 1, 0, 3, 3, 0, 2, 1, 0, 4, 1);
        assertEquals(2.3167546539234776, InformationTheory.entropy(a));
        assertEquals(2.316147658077609, InformationTheory.entropy(b));
    }
    
}

package org.tribuo.data.columnar.processors.response;

import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockMultiOutput;
import org.tribuo.test.MockMultiOutputFactory;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.util.Arrays;

public class ResponseProcessorRoundtripTest {

    @Test
    public void binaryTest() {
        BinaryResponseProcessor<MockMultiOutput> multiRespProc = new BinaryResponseProcessor<>(
                Arrays.asList("R1", "R2"),
                Arrays.asList("TRUE", "TRUE"),
                new MockMultiOutputFactory(),
                "true", "false", true);

        Helpers.testConfigurableRoundtrip(multiRespProc);

        BinaryResponseProcessor<MockOutput> singleRespProc = new BinaryResponseProcessor<>("R1", "TRUE", new MockOutputFactory());

        Helpers.testConfigurableRoundtrip(singleRespProc);
    }

    @Test
    public void fieldTest() {
        FieldResponseProcessor<MockMultiOutput> multiRespProc = new FieldResponseProcessor<>(
                Arrays.asList("R1", "R2"),
                Arrays.asList("A", "B"),
                new MockMultiOutputFactory(),
                true, false);

        Helpers.testConfigurableRoundtrip(multiRespProc);

        FieldResponseProcessor<MockOutput> singleRespProc = new FieldResponseProcessor<>("R1", "A", new MockOutputFactory());

        Helpers.testConfigurableRoundtrip(singleRespProc);
    }
}

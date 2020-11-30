package org.tribuo.data.columnar.processors.response;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

public class EmptyResponseProcessorTest {

    @Test
    public void basicTest() {
        MockOutputFactory outputFactory = new MockOutputFactory();
        EmptyResponseProcessor<MockOutput> rp = new EmptyResponseProcessor<>(outputFactory);

        // Check the output factory is stored correctly
        Assertions.assertEquals(outputFactory,rp.getOutputFactory());

        // Check the field name is right
        Assertions.assertEquals(EmptyResponseProcessor.FIELD_NAME, rp.getFieldName());

        // setFieldName is a no-op on this response processor
        rp.setFieldName("Something");
        Assertions.assertEquals(EmptyResponseProcessor.FIELD_NAME, rp.getFieldName());

        // Check that it doesn't throw exceptions when given odd text, and that it always returns Optional.empty.
        Assertions.assertFalse(rp.process("").isPresent());
        Assertions.assertFalse(rp.process("test").isPresent());
        Assertions.assertFalse(rp.process("!@$#$!").isPresent());
        Assertions.assertFalse(rp.process("\n").isPresent());
        Assertions.assertFalse(rp.process("\t").isPresent());
        Assertions.assertFalse(rp.process(null).isPresent());
    }

}

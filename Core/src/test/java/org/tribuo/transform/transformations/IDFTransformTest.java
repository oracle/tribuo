package org.tribuo.transform.transformations;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.IDFTransformerProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.transformations.IDFTransformation.IDFTransformer;

public class IDFTransformTest {

    
    @Test
    void testSerialize() throws Exception {
        IDFTransformer idft = new IDFTransformer(13, 17);
        TransformerProto tp = idft.serialize();
        assertEquals(0, tp.getVersion());
        assertEquals("org.tribuo.transform.transformations.IDFTransformation$IDFTransformer", tp.getClassName());
        IDFTransformerProto proto = tp.getSerializedData().unpack(IDFTransformerProto.class);
        assertEquals(13, proto.getDf());
        assertEquals(17, proto.getN());

        Transformer tD = ProtoUtil.deserialize(tp);
        assertEquals(idft, tD);
        
        assertEquals(TransformerProto.class, ProtoUtil.getSerializedClass(idft));

    }    

}

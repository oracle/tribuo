package org.tribuo.util;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.tribuo.hash.ModHashCodeHasher;
import org.tribuo.protos.core.HasherProto;
import org.tribuo.protos.core.ModHashCodeHasherProto;

public class ProtoUtilTest {

    @Test
    void testOld() throws Exception {
        ModHashCodeHasher hasher = new ModHashCodeHasher(200, "42");
        HasherProto hasherProto = hasher.serialize();
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.ModHashCodeHasher", hasherProto.getClassName());
        assertEquals(200, hasherProto.getSerializedData().unpack(ModHashCodeHasherProto.class).getDimension());
    }
    
    @Test
    void testSerialize() throws Exception {
        ModHashCodeHasher hasher = new ModHashCodeHasher(200, "42");
        HasherProto hasherProto = ProtoUtil.serialize(hasher);
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.ModHashCodeHasher", hasherProto.getClassName());
        ModHashCodeHasherProto proto = hasherProto.getSerializedData().unpack(ModHashCodeHasherProto.class);
        assertEquals(200, proto.getDimension());
    }
}

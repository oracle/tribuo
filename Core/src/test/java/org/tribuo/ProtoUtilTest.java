package org.tribuo;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.tribuo.hash.HashCodeHasher;
import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.hash.MessageDigestHasher;
import org.tribuo.hash.ModHashCodeHasher;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.CategoricalIDInfoProto;
import org.tribuo.protos.core.CategoricalInfoProto;
import org.tribuo.protos.core.FeatureDomainProto;
import org.tribuo.protos.core.HashedFeatureMapProto;
import org.tribuo.protos.core.HasherProto;
import org.tribuo.protos.core.MessageDigestHasherProto;
import org.tribuo.protos.core.ModHashCodeHasherProto;
import org.tribuo.protos.core.RealIDInfoProto;
import org.tribuo.protos.core.RealInfoProto;
import org.tribuo.protos.core.VariableInfoProto;

import com.google.protobuf.Message;

public class ProtoUtilTest {

    @Test
    void testHashedFeatureMap() throws Exception {
        MutableFeatureMap mfm = new MutableFeatureMap(); 
        mfm.add("goldrat", 1.618033988749);
        mfm.add("e", Math.E);
        mfm.add("pi", Math.PI);
        HashedFeatureMap hfm = HashedFeatureMap.generateHashedFeatureMap(mfm, new MessageDigestHasher("SHA-512", "abcdefghi"));
        FeatureDomainProto fdp = hfm.serialize();
        assertEquals(0, fdp.getVersion());
        assertEquals("org.tribuo.hash.HashedFeatureMap", fdp.getClassName());
        HashedFeatureMapProto hfmp = fdp.getSerializedData().unpack(HashedFeatureMapProto.class);
        HasherProto hasherProto = hfmp.getHasher();
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.MessageDigestHasher", hasherProto.getClassName());
        MessageDigestHasherProto mdhp = hasherProto.getSerializedData().unpack(MessageDigestHasherProto.class);
        assertEquals("SHA-512", mdhp.getHashType());
        
        HashedFeatureMap hfmD = ProtoUtil.deserialize(fdp);
        hfmD.setSalt("abcdefghi");
        assertEquals(hfm, hfmD);
    }
    
    @Test
    void testSerializeModHashCodeHasher() throws Exception {
        ModHashCodeHasher hasher = new ModHashCodeHasher(200, "abcdefghi");
        
        HasherProto hasherProto = hasher.serialize();
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.ModHashCodeHasher", hasherProto.getClassName());
        ModHashCodeHasherProto proto = hasherProto.getSerializedData().unpack(ModHashCodeHasherProto.class);
        assertEquals(200, proto.getDimension());

        ModHashCodeHasher hasherD = ProtoUtil.deserialize(hasherProto);
        hasherD.setSalt("abcdefghi");
        assertTrue(hasher.equals(hasherD));
        assertEquals(200, hasherProto.getSerializedData().unpack(ModHashCodeHasherProto.class).getDimension());
    }
    
    @Test
    void testMessageDigestHasher() throws Exception {
        MessageDigestHasher hasher = new MessageDigestHasher("SHA-256", "abcdefghi");
        HasherProto hasherProto = hasher.serialize();
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.MessageDigestHasher", hasherProto.getClassName());
        MessageDigestHasherProto proto = hasherProto.getSerializedData().unpack(MessageDigestHasherProto.class);
        assertEquals("SHA-256", proto.getHashType());
        
        MessageDigestHasher hasherD = ProtoUtil.deserialize(hasherProto);
        hasherD.setSalt("abcdefghi");
        assertEquals(hasher, hasherD);
    }

    @Test
    void testHashCodeHasher() throws Exception {
        HashCodeHasher hasher = new HashCodeHasher("abcdefghi");
        HasherProto hasherProto = hasher.serialize();
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.HashCodeHasher", hasherProto.getClassName());
        
        HashCodeHasher hasherD = ProtoUtil.deserialize(hasherProto);
        hasherD.setSalt("abcdefghi");
        assertEquals(hasher, hasherD);
     }

    
    @Test
    void testRealIDInfo() throws Exception {
        VariableInfo info = new RealIDInfo("bob", 100, 1000.0, 0.0, 25.0, 125.0, 12345);
        VariableInfoProto infoProto = info.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.RealIDInfo", infoProto.getClassName());
        RealIDInfoProto proto = infoProto.getSerializedData().unpack(RealIDInfoProto.class);
        assertEquals("bob", proto.getName());
        assertEquals(100, proto.getCount());
        assertEquals(1000.0, proto.getMax());
        assertEquals(0.0, proto.getMin());
        assertEquals(25.0, proto.getMean());
        assertEquals(125.0, proto.getSumSquares());
        assertEquals(12345, proto.getId());
        
        VariableInfo infoD = ProtoUtil.deserialize(infoProto);
        assertEquals(info, infoD);
    }
    
    @Test
    void testRealInfo() throws Exception {
        VariableInfo info = new RealInfo("bob", 100, 1000.0, 0.0, 25.0, 125.0);
        VariableInfoProto infoProto = info.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.RealInfo", infoProto.getClassName());
        RealInfoProto proto = infoProto.getSerializedData().unpack(RealInfoProto.class);
        assertEquals("bob", proto.getName());
        assertEquals(100, proto.getCount());
        assertEquals(1000.0, proto.getMax());
        assertEquals(0.0, proto.getMin());
        assertEquals(25.0, proto.getMean());
        assertEquals(125.0, proto.getSumSquares());
        
        VariableInfo infoD = ProtoUtil.deserialize(infoProto);
        assertEquals(info, infoD);
    }

    
    @Test
    void testCategoricalInfo() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                info.observe(i);
            });
        });
        
        VariableInfoProto infoProto = info.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalInfo", infoProto.getClassName());
        CategoricalInfoProto proto = infoProto.getSerializedData().unpack(CategoricalInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(90, proto.getCount());
        assertEquals(0, proto.getObservedCount());
        assertEquals(Double.NaN, proto.getObservedValue());
        
        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(9, keyList.size());
        assertEquals(9, valueList.size());
        
        Map<Double, Long> expectedCounts = new HashMap<>();
        IntStream.range(0, 10).forEach(i -> {
            long count = info.getObservationCount(i);
            expectedCounts.put((double)i, count);
        });
        
        for (int i=0; i<keyList.size(); i++) {
            assertEquals(expectedCounts.get(keyList.get(i)), valueList.get(i));
        }
        
        VariableInfo infoD = ProtoUtil.deserialize(infoProto);
        assertEquals(info, infoD);
    }

    @Test
    void testCategoricalInfo2() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            info.observe(5);
        });
        
        VariableInfoProto infoProto = info.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalInfo", infoProto.getClassName());
        CategoricalInfoProto proto = infoProto.getSerializedData().unpack(CategoricalInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(10, proto.getCount());
        
        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(0, keyList.size());
        assertEquals(0, valueList.size());
        assertEquals(5, proto.getObservedValue());
        assertEquals(10, proto.getObservedCount());
        
        VariableInfo infoD = ProtoUtil.deserialize(infoProto);
        assertEquals(info, infoD);
    }

    @Test
    void testCategoricalIdInfo() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                info.observe(i);
            });
        });

        CategoricalIDInfo idInfo = info.makeIDInfo(12345);

        VariableInfoProto infoProto = idInfo.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalIDInfo", infoProto.getClassName());
        CategoricalIDInfoProto proto = infoProto.getSerializedData().unpack(CategoricalIDInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(90, proto.getCount());
        assertEquals(12345, proto.getId());
        assertEquals(0, proto.getObservedCount());
        assertEquals(Double.NaN, proto.getObservedValue());
        
        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(keyList.size(), valueList.size());
        
        Map<Double, Long> expectedCounts = new HashMap<>();
        IntStream.range(0, 10).forEach(i -> {
            long count = idInfo.getObservationCount(i);
            expectedCounts.put((double)i, count);
        });
        
        for (int i=0; i<keyList.size(); i++) {
            assertEquals(expectedCounts.get(keyList.get(i)), valueList.get(i));
        }

        VariableInfo idInfoD = ProtoUtil.deserialize(infoProto);
        assertEquals(idInfo, idInfoD);

    }

    
    @Test
    void testGetSerializedClass() throws Exception {
        CategoricalInfo ci = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                ci.observe(i);
            });
        });

        assertEquals(VariableInfoProto.class, ProtoUtil.getSerializedClass(ci));
        CategoricalIDInfo cidi = ci.makeIDInfo(12345);
        assertEquals(VariableInfoProto.class, ProtoUtil.getSerializedClass(cidi));
        VariableInfo ridi = new RealIDInfo("bob", 100, 1000.0, 0.0, 25.0, 125.0, 12345);
        assertEquals(VariableInfoProto.class, ProtoUtil.getSerializedClass(ridi));
        RealInfo ri = new RealInfo("bob", 100, 1000.0, 0.0, 25.0, 125.0);
        assertEquals(VariableInfoProto.class, ProtoUtil.getSerializedClass(ri));
        
        MutableFeatureMap mfm = new MutableFeatureMap(); 
        mfm.add("goldrat", 1.618033988749);
        mfm.add("e", Math.E);
        mfm.add("pi", Math.PI);
        HashedFeatureMap hfm = HashedFeatureMap.generateHashedFeatureMap(mfm, new MessageDigestHasher("SHA-512", "abcdefghi"));
        assertEquals(FeatureDomainProto.class, ProtoUtil.getSerializedClass(hfm));
        
        ModHashCodeHasher mdch = new ModHashCodeHasher(200, "abcdefghi");
        assertEquals(HasherProto.class, ProtoUtil.getSerializedClass(mdch));

        MessageDigestHasher mdh = new MessageDigestHasher("SHA-256", "abcdefghi");
        assertEquals(HasherProto.class, ProtoUtil.getSerializedClass(mdh));

        HashCodeHasher hch = new HashCodeHasher("abcdefghi");
        assertEquals(HasherProto.class, ProtoUtil.getSerializedClass(hch));

        assertEquals(CategoricalIDInfoProto.class, ProtoUtil.getSerializedClass(new PSC()));
        assertEquals(RealIDInfoProto.class, ProtoUtil.getSerializedClass(new PSD2()));
        assertEquals(RealIDInfoProto.class, ProtoUtil.getSerializedClass(new PSC2()));
        assertThrows(IllegalArgumentException.class, () -> ProtoUtil.getSerializedClass(new PSB2<RealInfoProto>()));
    }


    
    public static interface IPS<W, X, Y extends Message> extends ProtoSerializable<Y>{
        
    }
    
    public static class PSA<A, B extends Message> implements IPS<String, String, B>{
        
    }
    
    public static class PSB<C extends Message> extends PSA<String, C>{
        
    }

    public static class PSC extends PSB<CategoricalIDInfoProto>{
        
    }

    public static interface IPS2<Y extends Message> extends ProtoSerializable<Y>{
        
    }

    //Tricky!  we purposefully mixed up the type variable names
    public static class PSA2<A, B extends Message, Y extends Message> implements IPS2<B>{
        
    }
    
    public static class PSB2<Y extends Message> extends PSA2<String, Y, CategoricalIDInfoProto>{
        
    }

    public static class PSC2 extends PSB<RealIDInfoProto>{
        
    }

    public static class PSD2 extends PSC2{
        
    }
    
}
// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-test.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.test.protos;

public final class TribuoCoreTest {
  private TribuoCoreTest() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_TestCountTransformerProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_TestCountTransformerProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_MockOutputProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_MockOutputProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_MockMultiOutputProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_MockMultiOutputProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_MockOutputInfoProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_MockOutputInfoProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_MockMultiOutputInfoProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_MockMultiOutputInfoProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_MockModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_MockModelProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\026tribuo-core-test.proto\022\013tribuo.core\032\021t" +
      "ribuo-core.proto\032\021olcut_proto.proto\"m\n\031T" +
      "estCountTransformerProto\022\r\n\005count\030\001 \001(\005\022" +
      "\023\n\013sparseCount\030\002 \001(\005\022\024\n\014countMapKeys\030\003 \003" +
      "(\001\022\026\n\016countMapValues\030\004 \003(\003\" \n\017MockOutput" +
      "Proto\022\r\n\005label\030\001 \001(\t\"4\n\024MockMultiOutputP" +
      "roto\022\r\n\005label\030\001 \003(\t\022\r\n\005score\030\002 \001(\001\"l\n\023Mo" +
      "ckOutputInfoProto\022\r\n\005label\030\001 \003(\t\022\016\n\006coun" +
      "ts\030\002 \003(\003\022\n\n\002id\030\003 \003(\005\022\024\n\014unknownCount\030\004 \001" +
      "(\005\022\024\n\014labelCounter\030\005 \001(\005\"q\n\030MockMultiOut" +
      "putInfoProto\022\r\n\005label\030\001 \003(\t\022\016\n\006counts\030\002 " +
      "\003(\003\022\n\n\002id\030\003 \003(\005\022\024\n\014unknownCount\030\004 \001(\005\022\024\n" +
      "\014labelCounter\030\005 \001(\005\"W\n\016MockModelProto\022-\n" +
      "\010metadata\030\001 \001(\0132\033.tribuo.core.ModelDataP" +
      "roto\022\026\n\016constantOutput\030\002 \001(\tB\032\n\026org.trib" +
      "uo.test.protosP\001b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
          com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor(),
        });
    internal_static_tribuo_core_TestCountTransformerProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_core_TestCountTransformerProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_TestCountTransformerProto_descriptor,
        new java.lang.String[] { "Count", "SparseCount", "CountMapKeys", "CountMapValues", });
    internal_static_tribuo_core_MockOutputProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_core_MockOutputProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_MockOutputProto_descriptor,
        new java.lang.String[] { "Label", });
    internal_static_tribuo_core_MockMultiOutputProto_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_tribuo_core_MockMultiOutputProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_MockMultiOutputProto_descriptor,
        new java.lang.String[] { "Label", "Score", });
    internal_static_tribuo_core_MockOutputInfoProto_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_tribuo_core_MockOutputInfoProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_MockOutputInfoProto_descriptor,
        new java.lang.String[] { "Label", "Counts", "Id", "UnknownCount", "LabelCounter", });
    internal_static_tribuo_core_MockMultiOutputInfoProto_descriptor =
      getDescriptor().getMessageTypes().get(4);
    internal_static_tribuo_core_MockMultiOutputInfoProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_MockMultiOutputInfoProto_descriptor,
        new java.lang.String[] { "Label", "Counts", "Id", "UnknownCount", "LabelCounter", });
    internal_static_tribuo_core_MockModelProto_descriptor =
      getDescriptor().getMessageTypes().get(5);
    internal_static_tribuo_core_MockModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_MockModelProto_descriptor,
        new java.lang.String[] { "Metadata", "ConstantOutput", });
    org.tribuo.protos.core.TribuoCore.getDescriptor();
    com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

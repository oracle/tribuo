// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-math.proto

package org.tribuo.math.protos;

public final class TribuoMath {
  private TribuoMath() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_KernelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_KernelProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_MergerProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_MergerProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_NeighbourFactoryProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_NeighbourFactoryProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_NormalizerProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_NormalizerProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_ParametersProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_ParametersProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_TensorProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_TensorProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\021tribuo-math.proto\022\013tribuo.core\032\031google" +
      "/protobuf/any.proto\032\021tribuo-core.proto\032\021" +
      "olcut_proto.proto\"a\n\013KernelProto\022\017\n\007vers" +
      "ion\030\001 \001(\005\022\022\n\nclass_name\030\002 \001(\t\022-\n\017seriali" +
      "zed_data\030\003 \001(\0132\024.google.protobuf.Any\"a\n\013" +
      "MergerProto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass_na" +
      "me\030\002 \001(\t\022-\n\017serialized_data\030\003 \001(\0132\024.goog" +
      "le.protobuf.Any\"k\n\025NeighbourFactoryProto" +
      "\022\017\n\007version\030\001 \001(\005\022\022\n\nclass_name\030\002 \001(\t\022-\n" +
      "\017serialized_data\030\003 \001(\0132\024.google.protobuf" +
      ".Any\"e\n\017NormalizerProto\022\017\n\007version\030\001 \001(\005" +
      "\022\022\n\nclass_name\030\002 \001(\t\022-\n\017serialized_data\030" +
      "\003 \001(\0132\024.google.protobuf.Any\"e\n\017Parameter" +
      "sProto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass_name\030\002 " +
      "\001(\t\022-\n\017serialized_data\030\003 \001(\0132\024.google.pr" +
      "otobuf.Any\"a\n\013TensorProto\022\017\n\007version\030\001 \001" +
      "(\005\022\022\n\nclass_name\030\002 \001(\t\022-\n\017serialized_dat" +
      "a\030\003 \001(\0132\024.google.protobuf.AnyB\032\n\026org.tri" +
      "buo.math.protosP\001b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          com.google.protobuf.AnyProto.getDescriptor(),
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
          com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor(),
        });
    internal_static_tribuo_core_KernelProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_core_KernelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_KernelProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_MergerProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_core_MergerProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_MergerProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_NeighbourFactoryProto_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_tribuo_core_NeighbourFactoryProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_NeighbourFactoryProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_NormalizerProto_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_tribuo_core_NormalizerProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_NormalizerProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_ParametersProto_descriptor =
      getDescriptor().getMessageTypes().get(4);
    internal_static_tribuo_core_ParametersProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_ParametersProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_TensorProto_descriptor =
      getDescriptor().getMessageTypes().get(5);
    internal_static_tribuo_core_TensorProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_TensorProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    com.google.protobuf.AnyProto.getDescriptor();
    org.tribuo.protos.core.TribuoCore.getDescriptor();
    com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

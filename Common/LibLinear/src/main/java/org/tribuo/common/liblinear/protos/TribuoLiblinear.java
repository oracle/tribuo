// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-liblinear.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.common.liblinear.protos;

public final class TribuoLiblinear {
  private TribuoLiblinear() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_common_liblinear_LibLinearProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_common_liblinear_LibLinearProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_common_liblinear_LibLinearModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_common_liblinear_LibLinearModelProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\026tribuo-liblinear.proto\022\027tribuo.common." +
      "liblinear\032\021tribuo-core.proto\"\200\001\n\016LibLine" +
      "arProto\022\014\n\004bias\030\001 \001(\001\022\r\n\005label\030\002 \003(\005\022\020\n\010" +
      "nr_class\030\003 \001(\005\022\022\n\nnr_feature\030\004 \001(\005\022\023\n\013so" +
      "lver_type\030\005 \001(\t\022\t\n\001w\030\006 \003(\001\022\013\n\003rho\030\007 \001(\001\"" +
      "T\n\023LibLinearModelProto\022-\n\010metadata\030\001 \001(\013" +
      "2\033.tribuo.core.ModelDataProto\022\016\n\006models\030" +
      "\002 \003(\014B&\n\"org.tribuo.common.liblinear.pro" +
      "tosP\001b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
        });
    internal_static_tribuo_common_liblinear_LibLinearProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_common_liblinear_LibLinearProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_common_liblinear_LibLinearProto_descriptor,
        new java.lang.String[] { "Bias", "Label", "NrClass", "NrFeature", "SolverType", "W", "Rho", });
    internal_static_tribuo_common_liblinear_LibLinearModelProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_common_liblinear_LibLinearModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_common_liblinear_LibLinearModelProto_descriptor,
        new java.lang.String[] { "Metadata", "Models", });
    org.tribuo.protos.core.TribuoCore.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

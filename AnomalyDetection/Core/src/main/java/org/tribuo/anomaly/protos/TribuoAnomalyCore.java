// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-anomaly-core.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.anomaly.protos;

public final class TribuoAnomalyCore {
  private TribuoAnomalyCore() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_anomaly_EventProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_anomaly_EventProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_anomaly_AnomalyInfoProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_anomaly_AnomalyInfoProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\031tribuo-anomaly-core.proto\022\016tribuo.anom" +
      "aly\032\021tribuo-core.proto\032\021olcut_proto.prot" +
      "o\"\220\001\n\nEventProto\0223\n\005event\030\001 \001(\0162$.tribuo" +
      ".anomaly.EventProto.EventType\022\r\n\005score\030\002" +
      " \001(\001\">\n\tEventType\022\014\n\010EXPECTED\020\000\022\r\n\tANOMA" +
      "LOUS\020\001\022\024\n\007UNKNOWN\020\377\377\377\377\377\377\377\377\377\001\"U\n\020AnomalyI" +
      "nfoProto\022\025\n\rexpectedCount\030\001 \001(\003\022\024\n\014anoma" +
      "lyCount\030\002 \001(\003\022\024\n\014unknownCount\030\003 \001(\005B\035\n\031o" +
      "rg.tribuo.anomaly.protosP\001b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
          com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor(),
        });
    internal_static_tribuo_anomaly_EventProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_anomaly_EventProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_anomaly_EventProto_descriptor,
        new java.lang.String[] { "Event", "Score", });
    internal_static_tribuo_anomaly_AnomalyInfoProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_anomaly_AnomalyInfoProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_anomaly_AnomalyInfoProto_descriptor,
        new java.lang.String[] { "ExpectedCount", "AnomalyCount", "UnknownCount", });
    org.tribuo.protos.core.TribuoCore.getDescriptor();
    com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

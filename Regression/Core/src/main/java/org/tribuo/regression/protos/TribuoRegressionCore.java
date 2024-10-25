// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-regression-core.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.regression.protos;

public final class TribuoRegressionCore {
  private TribuoRegressionCore() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_regression_RegressorProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_regression_RegressorProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_regression_DimensionTupleProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_regression_DimensionTupleProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_regression_RegressionFactoryProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_regression_RegressionFactoryProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_regression_MutableRegressionInfoProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_regression_MutableRegressionInfoProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_regression_ImmutableRegressionInfoProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_regression_ImmutableRegressionInfoProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_regression_DummyRegressionModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_regression_DummyRegressionModelProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\034tribuo-regression-core.proto\022\021tribuo.r" +
      "egression\032\021tribuo-core.proto\"?\n\016Regresso" +
      "rProto\022\014\n\004name\030\001 \003(\t\022\r\n\005value\030\002 \003(\001\022\020\n\010v" +
      "ariance\030\003 \003(\001\"D\n\023DimensionTupleProto\022\014\n\004" +
      "name\030\001 \001(\t\022\r\n\005value\030\002 \001(\001\022\020\n\010variance\030\003 " +
      "\001(\001\"+\n\026RegressionFactoryProto\022\021\n\tsplitCh" +
      "ar\030\001 \001(\t\"\242\001\n\032MutableRegressionInfoProto\022" +
      "\r\n\005label\030\001 \003(\t\022\r\n\005count\030\002 \003(\003\022\013\n\003max\030\003 \003" +
      "(\001\022\013\n\003min\030\004 \003(\001\022\014\n\004mean\030\005 \003(\001\022\022\n\nsumSqua" +
      "res\030\006 \003(\001\022\024\n\014unknownCount\030\007 \001(\005\022\024\n\014overa" +
      "llCount\030\010 \001(\003\"\260\001\n\034ImmutableRegressionInf" +
      "oProto\022\r\n\005label\030\001 \003(\t\022\r\n\005count\030\002 \003(\003\022\n\n\002" +
      "id\030\003 \003(\005\022\013\n\003max\030\004 \003(\001\022\013\n\003min\030\005 \003(\001\022\014\n\004me" +
      "an\030\006 \003(\001\022\022\n\nsumSquares\030\007 \003(\001\022\024\n\014unknownC" +
      "ount\030\010 \001(\005\022\024\n\014overallCount\030\t \001(\003\"\321\001\n\031Dum" +
      "myRegressionModelProto\022-\n\010metadata\030\001 \001(\013" +
      "2\033.tribuo.core.ModelDataProto\022\022\n\ndummy_t" +
      "ype\030\002 \001(\t\022(\n\006output\030\003 \001(\0132\030.tribuo.core." +
      "OutputProto\022\014\n\004seed\030\004 \001(\003\022\r\n\005means\030\005 \003(\001" +
      "\022\021\n\tvariances\030\006 \003(\001\022\027\n\017dimension_names\030\007" +
      " \003(\tB \n\034org.tribuo.regression.protosP\001b\006" +
      "proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
        });
    internal_static_tribuo_regression_RegressorProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_regression_RegressorProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_regression_RegressorProto_descriptor,
        new java.lang.String[] { "Name", "Value", "Variance", });
    internal_static_tribuo_regression_DimensionTupleProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_regression_DimensionTupleProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_regression_DimensionTupleProto_descriptor,
        new java.lang.String[] { "Name", "Value", "Variance", });
    internal_static_tribuo_regression_RegressionFactoryProto_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_tribuo_regression_RegressionFactoryProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_regression_RegressionFactoryProto_descriptor,
        new java.lang.String[] { "SplitChar", });
    internal_static_tribuo_regression_MutableRegressionInfoProto_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_tribuo_regression_MutableRegressionInfoProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_regression_MutableRegressionInfoProto_descriptor,
        new java.lang.String[] { "Label", "Count", "Max", "Min", "Mean", "SumSquares", "UnknownCount", "OverallCount", });
    internal_static_tribuo_regression_ImmutableRegressionInfoProto_descriptor =
      getDescriptor().getMessageTypes().get(4);
    internal_static_tribuo_regression_ImmutableRegressionInfoProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_regression_ImmutableRegressionInfoProto_descriptor,
        new java.lang.String[] { "Label", "Count", "Id", "Max", "Min", "Mean", "SumSquares", "UnknownCount", "OverallCount", });
    internal_static_tribuo_regression_DummyRegressionModelProto_descriptor =
      getDescriptor().getMessageTypes().get(5);
    internal_static_tribuo_regression_DummyRegressionModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_regression_DummyRegressionModelProto_descriptor,
        new java.lang.String[] { "Metadata", "DummyType", "Output", "Seed", "Means", "Variances", "DimensionNames", });
    org.tribuo.protos.core.TribuoCore.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

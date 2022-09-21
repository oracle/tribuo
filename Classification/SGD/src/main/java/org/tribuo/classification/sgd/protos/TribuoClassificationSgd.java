// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-classification-sgd.proto

package org.tribuo.classification.sgd.protos;

public final class TribuoClassificationSgd {
  private TribuoClassificationSgd() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_classification_sgd_CRFParametersProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_classification_sgd_CRFParametersProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_classification_sgd_CRFModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_classification_sgd_CRFModelProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_classification_sgd_FMClassificationModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_classification_sgd_FMClassificationModelProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_classification_sgd_KernelSVMModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_classification_sgd_KernelSVMModelProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\037tribuo-classification-sgd.proto\022\031tribu" +
      "o.classification.sgd\032\021tribuo-core.proto\032" +
      "\021tribuo-math.proto\"\322\001\n\022CRFParametersProt" +
      "o\022\023\n\013numFeatures\030\001 \001(\005\022\021\n\tnumLabels\030\002 \001(" +
      "\005\022(\n\006biases\030\003 \001(\0132\030.tribuo.math.TensorPr" +
      "oto\0225\n\023featureLabelWeights\030\004 \001(\0132\030.tribu" +
      "o.math.TensorProto\0223\n\021labelLabelWeights\030" +
      "\005 \001(\0132\030.tribuo.math.TensorProto\"\205\001\n\rCRFM" +
      "odelProto\022-\n\010metadata\030\001 \001(\0132\033.tribuo.cor" +
      "e.ModelDataProto\022,\n\006params\030\002 \001(\0132\034.tribu" +
      "o.math.ParametersProto\022\027\n\017confidence_typ" +
      "e\030\003 \001(\t\"\253\001\n\032FMClassificationModelProto\022-" +
      "\n\010metadata\030\001 \001(\0132\033.tribuo.core.ModelData" +
      "Proto\022,\n\006params\030\002 \001(\0132\034.tribuo.math.Para" +
      "metersProto\0220\n\nnormalizer\030\003 \001(\0132\034.tribuo" +
      ".math.NormalizerProto\"\255\001\n\034Classification" +
      "LinearSGDProto\022-\n\010metadata\030\001 \001(\0132\033.tribu" +
      "o.core.ModelDataProto\022,\n\006params\030\002 \001(\0132\034." +
      "tribuo.math.ParametersProto\0220\n\nnormalize" +
      "r\030\003 \001(\0132\034.tribuo.math.NormalizerProto\"\314\001" +
      "\n\023KernelSVMModelProto\022-\n\010metadata\030\001 \001(\0132" +
      "\033.tribuo.core.ModelDataProto\022(\n\006kernel\030\002" +
      " \001(\0132\030.tribuo.math.KernelProto\022)\n\007weight" +
      "s\030\003 \001(\0132\030.tribuo.math.TensorProto\0221\n\017sup" +
      "port_vectors\030\004 \003(\0132\030.tribuo.math.TensorP" +
      "rotoB(\n$org.tribuo.classification.sgd.pr" +
      "otosP\001b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
          org.tribuo.math.protos.TribuoMath.getDescriptor(),
        });
    internal_static_tribuo_classification_sgd_CRFParametersProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_classification_sgd_CRFParametersProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_classification_sgd_CRFParametersProto_descriptor,
        new java.lang.String[] { "NumFeatures", "NumLabels", "Biases", "FeatureLabelWeights", "LabelLabelWeights", });
    internal_static_tribuo_classification_sgd_CRFModelProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_classification_sgd_CRFModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_classification_sgd_CRFModelProto_descriptor,
        new java.lang.String[] { "Metadata", "Params", "ConfidenceType", });
    internal_static_tribuo_classification_sgd_FMClassificationModelProto_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_tribuo_classification_sgd_FMClassificationModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_classification_sgd_FMClassificationModelProto_descriptor,
        new java.lang.String[] { "Metadata", "Params", "Normalizer", });
    internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_descriptor,
        new java.lang.String[] { "Metadata", "Params", "Normalizer", });
    internal_static_tribuo_classification_sgd_KernelSVMModelProto_descriptor =
      getDescriptor().getMessageTypes().get(4);
    internal_static_tribuo_classification_sgd_KernelSVMModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_classification_sgd_KernelSVMModelProto_descriptor,
        new java.lang.String[] { "Metadata", "Kernel", "Weights", "SupportVectors", });
    org.tribuo.protos.core.TribuoCore.getDescriptor();
    org.tribuo.math.protos.TribuoMath.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

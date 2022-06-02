// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core.proto

package org.tribuo.protos.core;

public final class TribuoCore {
  private TribuoCore() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_ModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_ModelProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_OutputFactoryProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_OutputFactoryProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_FeatureDomainProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_FeatureDomainProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_VariableInfoProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_VariableInfoProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_OutputDomainProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_OutputDomainProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_ModelDataProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_ModelDataProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_OutputProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_OutputProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_ExampleProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_ExampleProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_ExampleProto_MetadataEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_ExampleProto_MetadataEntry_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_DatasetProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_DatasetProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_PredictionProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_PredictionProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_PredictionProto_OutputScoresEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_PredictionProto_OutputScoresEntry_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_EnsembleCombinerProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_EnsembleCombinerProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_TransformerProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_TransformerProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_TransformerListProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_TransformerListProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_TransformerMapProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_TransformerMapProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_TransformerMapProto_TransformersEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_TransformerMapProto_TransformersEntry_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_HasherProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_HasherProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_core_MeanVarianceProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_core_MeanVarianceProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\021tribuo-core.proto\022\013tribuo.core\032\031google" +
      "/protobuf/any.proto\032\021olcut_proto.proto\"\234" +
      "\002\n\nModelProto\022\017\n\007version\030\001 \001(\005\022\014\n\004name\030\002" +
      " \001(\t\022.\n\nprovenance\030\003 \001(\0132\032.olcut.RootPro" +
      "venanceProto\022\036\n\026generate_probabilities\030\004" +
      " \001(\010\0227\n\016feature_domain\030\005 \001(\0132\037.tribuo.co" +
      "re.FeatureDomainProto\0225\n\routput_domain\030\006" +
      " \001(\0132\036.tribuo.core.OutputDomainProto\022/\n\n" +
      "model_data\030\007 \001(\0132\033.tribuo.core.ModelData" +
      "Proto\"h\n\022OutputFactoryProto\022\017\n\007version\030\001" +
      " \001(\005\022\022\n\nclass_name\030\002 \001(\t\022-\n\017serialized_d" +
      "ata\030\003 \001(\0132\024.google.protobuf.Any\"h\n\022Featu" +
      "reDomainProto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass_" +
      "name\030\002 \001(\t\022-\n\017serialized_data\030\003 \001(\0132\024.go" +
      "ogle.protobuf.Any\"g\n\021VariableInfoProto\022\017" +
      "\n\007version\030\001 \001(\005\022\022\n\nclass_name\030\002 \001(\t\022-\n\017s" +
      "erialized_data\030\003 \001(\0132\024.google.protobuf.A" +
      "ny\"g\n\021OutputDomainProto\022\017\n\007version\030\001 \001(\005" +
      "\022\022\n\nclass_name\030\002 \001(\t\022-\n\017serialized_data\030" +
      "\003 \001(\0132\024.google.protobuf.Any\"d\n\016ModelData" +
      "Proto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass_name\030\002 \001" +
      "(\t\022-\n\017serialized_data\030\003 \001(\0132\024.google.pro" +
      "tobuf.Any\"a\n\013OutputProto\022\017\n\007version\030\001 \001(" +
      "\005\022\022\n\nclass_name\030\002 \001(\t\022-\n\017serialized_data" +
      "\030\003 \001(\0132\024.google.protobuf.Any\"\366\001\n\014Example" +
      "Proto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass_name\030\002 \001" +
      "(\t\022(\n\006output\030\003 \001(\0132\030.tribuo.core.OutputP" +
      "roto\022\024\n\014feature_name\030\004 \003(\t\022\025\n\rfeature_va" +
      "lue\030\005 \003(\001\0229\n\010metadata\030\006 \003(\0132\'.tribuo.cor" +
      "e.ExampleProto.MetadataEntry\032/\n\rMetadata" +
      "Entry\022\013\n\003key\030\001 \001(\t\022\r\n\005value\030\002 \001(\t:\0028\001\"\272\002" +
      "\n\014DatasetProto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass" +
      "_name\030\002 \001(\t\022.\n\nprovenance\030\003 \001(\0132\032.olcut." +
      "RootProvenanceProto\0227\n\016feature_domain\030\004 " +
      "\001(\0132\037.tribuo.core.FeatureDomainProto\0225\n\r" +
      "output_domain\030\005 \001(\0132\036.tribuo.core.Output" +
      "DomainProto\022+\n\010examples\030\006 \003(\0132\031.tribuo.c" +
      "ore.ExampleProto\0228\n\024transform_provenance" +
      "\030\007 \001(\0132\032.olcut.ListProvenanceProto\"\337\002\n\017P" +
      "redictionProto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass" +
      "_name\030\002 \001(\t\022*\n\007example\030\003 \001(\0132\031.tribuo.co" +
      "re.ExampleProto\022(\n\006output\030\004 \001(\0132\030.tribuo" +
      ".core.OutputProto\022\023\n\013probability\030\005 \001(\010\022\020" +
      "\n\010num_used\030\006 \001(\005\022\024\n\014example_size\030\007 \001(\005\022E" +
      "\n\routput_scores\030\010 \003(\0132..tribuo.core.Pred" +
      "ictionProto.OutputScoresEntry\032M\n\021OutputS" +
      "coresEntry\022\013\n\003key\030\001 \001(\t\022\'\n\005value\030\002 \001(\0132\030" +
      ".tribuo.core.OutputProto:\0028\001\"k\n\025Ensemble" +
      "CombinerProto\022\017\n\007version\030\001 \001(\005\022\022\n\nclass_" +
      "name\030\002 \001(\t\022-\n\017serialized_data\030\003 \001(\0132\024.go" +
      "ogle.protobuf.Any\"f\n\020TransformerProto\022\017\n" +
      "\007version\030\001 \001(\005\022\022\n\nclass_name\030\002 \001(\t\022-\n\017se" +
      "rialized_data\030\003 \001(\0132\024.google.protobuf.An" +
      "y\"J\n\024TransformerListProto\0222\n\013transformer" +
      "\030\001 \003(\0132\035.tribuo.core.TransformerProto\"\300\002" +
      "\n\023TransformerMapProto\022\017\n\007version\030\001 \001(\005\022H" +
      "\n\014transformers\030\002 \003(\01322.tribuo.core.Trans" +
      "formerMapProto.TransformersEntry\0225\n\021data" +
      "setProvenance\030\003 \001(\0132\032.olcut.RootProvenan" +
      "ceProto\022?\n\033transformationMapProvenance\030\004" +
      " \001(\0132\032.olcut.RootProvenanceProto\032V\n\021Tran" +
      "sformersEntry\022\013\n\003key\030\001 \001(\t\0220\n\005value\030\002 \001(" +
      "\0132!.tribuo.core.TransformerListProto:\0028\001" +
      "\"a\n\013HasherProto\022\017\n\007version\030\001 \001(\005\022\022\n\nclas" +
      "s_name\030\002 \001(\t\022-\n\017serialized_data\030\003 \001(\0132\024." +
      "google.protobuf.Any\"o\n\021MeanVarianceProto" +
      "\022\017\n\007version\030\001 \001(\005\022\013\n\003max\030\002 \001(\001\022\013\n\003min\030\003 " +
      "\001(\001\022\014\n\004mean\030\004 \001(\001\022\022\n\nsumSquares\030\005 \001(\001\022\r\n" +
      "\005count\030\006 \001(\003B\032\n\026org.tribuo.protos.coreP\001" +
      "b\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          com.google.protobuf.AnyProto.getDescriptor(),
          com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor(),
        });
    internal_static_tribuo_core_ModelProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_core_ModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_ModelProto_descriptor,
        new java.lang.String[] { "Version", "Name", "Provenance", "GenerateProbabilities", "FeatureDomain", "OutputDomain", "ModelData", });
    internal_static_tribuo_core_OutputFactoryProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_core_OutputFactoryProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_OutputFactoryProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_FeatureDomainProto_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_tribuo_core_FeatureDomainProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_FeatureDomainProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_VariableInfoProto_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_tribuo_core_VariableInfoProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_VariableInfoProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_OutputDomainProto_descriptor =
      getDescriptor().getMessageTypes().get(4);
    internal_static_tribuo_core_OutputDomainProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_OutputDomainProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_ModelDataProto_descriptor =
      getDescriptor().getMessageTypes().get(5);
    internal_static_tribuo_core_ModelDataProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_ModelDataProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_OutputProto_descriptor =
      getDescriptor().getMessageTypes().get(6);
    internal_static_tribuo_core_OutputProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_OutputProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_ExampleProto_descriptor =
      getDescriptor().getMessageTypes().get(7);
    internal_static_tribuo_core_ExampleProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_ExampleProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "Output", "FeatureName", "FeatureValue", "Metadata", });
    internal_static_tribuo_core_ExampleProto_MetadataEntry_descriptor =
      internal_static_tribuo_core_ExampleProto_descriptor.getNestedTypes().get(0);
    internal_static_tribuo_core_ExampleProto_MetadataEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_ExampleProto_MetadataEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    internal_static_tribuo_core_DatasetProto_descriptor =
      getDescriptor().getMessageTypes().get(8);
    internal_static_tribuo_core_DatasetProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_DatasetProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "Provenance", "FeatureDomain", "OutputDomain", "Examples", "TransformProvenance", });
    internal_static_tribuo_core_PredictionProto_descriptor =
      getDescriptor().getMessageTypes().get(9);
    internal_static_tribuo_core_PredictionProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_PredictionProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "Example", "Output", "Probability", "NumUsed", "ExampleSize", "OutputScores", });
    internal_static_tribuo_core_PredictionProto_OutputScoresEntry_descriptor =
      internal_static_tribuo_core_PredictionProto_descriptor.getNestedTypes().get(0);
    internal_static_tribuo_core_PredictionProto_OutputScoresEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_PredictionProto_OutputScoresEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    internal_static_tribuo_core_EnsembleCombinerProto_descriptor =
      getDescriptor().getMessageTypes().get(10);
    internal_static_tribuo_core_EnsembleCombinerProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_EnsembleCombinerProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_TransformerProto_descriptor =
      getDescriptor().getMessageTypes().get(11);
    internal_static_tribuo_core_TransformerProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_TransformerProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_TransformerListProto_descriptor =
      getDescriptor().getMessageTypes().get(12);
    internal_static_tribuo_core_TransformerListProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_TransformerListProto_descriptor,
        new java.lang.String[] { "Transformer", });
    internal_static_tribuo_core_TransformerMapProto_descriptor =
      getDescriptor().getMessageTypes().get(13);
    internal_static_tribuo_core_TransformerMapProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_TransformerMapProto_descriptor,
        new java.lang.String[] { "Version", "Transformers", "DatasetProvenance", "TransformationMapProvenance", });
    internal_static_tribuo_core_TransformerMapProto_TransformersEntry_descriptor =
      internal_static_tribuo_core_TransformerMapProto_descriptor.getNestedTypes().get(0);
    internal_static_tribuo_core_TransformerMapProto_TransformersEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_TransformerMapProto_TransformersEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    internal_static_tribuo_core_HasherProto_descriptor =
      getDescriptor().getMessageTypes().get(14);
    internal_static_tribuo_core_HasherProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_HasherProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_core_MeanVarianceProto_descriptor =
      getDescriptor().getMessageTypes().get(15);
    internal_static_tribuo_core_MeanVarianceProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_core_MeanVarianceProto_descriptor,
        new java.lang.String[] { "Version", "Max", "Min", "Mean", "SumSquares", "Count", });
    com.google.protobuf.AnyProto.getDescriptor();
    com.oracle.labs.mlrg.olcut.config.protobuf.protos.OlcutProto.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

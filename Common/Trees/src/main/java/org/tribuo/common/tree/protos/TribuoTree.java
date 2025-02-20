// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-tree.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.common.tree.protos;

public final class TribuoTree {
  private TribuoTree() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_common_tree_TreeNodeProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_common_tree_TreeNodeProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_common_tree_SplitNodeProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_common_tree_SplitNodeProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_common_tree_LeafNodeProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_common_tree_LeafNodeProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_common_tree_LeafNodeProto_ScoreEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_common_tree_LeafNodeProto_ScoreEntry_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tribuo_common_tree_TreeModelProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tribuo_common_tree_TreeModelProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\021tribuo-tree.proto\022\022tribuo.common.tree\032" +
      "\031google/protobuf/any.proto\032\021tribuo-core." +
      "proto\"c\n\rTreeNodeProto\022\017\n\007version\030\001 \001(\005\022" +
      "\022\n\nclass_name\030\002 \001(\t\022-\n\017serialized_data\030\003" +
      " \001(\0132\024.google.protobuf.Any\"\261\001\n\016SplitNode" +
      "Proto\022\022\n\nparent_idx\030\001 \001(\005\022\017\n\007cur_idx\030\002 \001" +
      "(\005\022\030\n\020greater_than_idx\030\003 \001(\005\022\036\n\026less_tha" +
      "n_or_equal_idx\030\004 \001(\005\022\031\n\021split_feature_id" +
      "x\030\005 \001(\005\022\023\n\013split_value\030\006 \001(\001\022\020\n\010impurity" +
      "\030\007 \001(\001\"\226\002\n\rLeafNodeProto\022\022\n\nparent_idx\030\001" +
      " \001(\005\022\017\n\007cur_idx\030\002 \001(\005\022\020\n\010impurity\030\003 \001(\001\022" +
      "(\n\006output\030\004 \001(\0132\030.tribuo.core.OutputProt" +
      "o\022;\n\005score\030\005 \003(\0132,.tribuo.common.tree.Le" +
      "afNodeProto.ScoreEntry\022\037\n\027generates_prob" +
      "abilities\030\006 \001(\010\032F\n\nScoreEntry\022\013\n\003key\030\001 \001" +
      "(\t\022\'\n\005value\030\002 \001(\0132\030.tribuo.core.OutputPr" +
      "oto:\0028\001\"q\n\016TreeModelProto\022-\n\010metadata\030\001 " +
      "\001(\0132\033.tribuo.core.ModelDataProto\0220\n\005node" +
      "s\030\002 \003(\0132!.tribuo.common.tree.TreeNodePro" +
      "toB!\n\035org.tribuo.common.tree.protosP\001b\006p" +
      "roto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          com.google.protobuf.AnyProto.getDescriptor(),
          org.tribuo.protos.core.TribuoCore.getDescriptor(),
        });
    internal_static_tribuo_common_tree_TreeNodeProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tribuo_common_tree_TreeNodeProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_common_tree_TreeNodeProto_descriptor,
        new java.lang.String[] { "Version", "ClassName", "SerializedData", });
    internal_static_tribuo_common_tree_SplitNodeProto_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_tribuo_common_tree_SplitNodeProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_common_tree_SplitNodeProto_descriptor,
        new java.lang.String[] { "ParentIdx", "CurIdx", "GreaterThanIdx", "LessThanOrEqualIdx", "SplitFeatureIdx", "SplitValue", "Impurity", });
    internal_static_tribuo_common_tree_LeafNodeProto_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_tribuo_common_tree_LeafNodeProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_common_tree_LeafNodeProto_descriptor,
        new java.lang.String[] { "ParentIdx", "CurIdx", "Impurity", "Output", "Score", "GeneratesProbabilities", });
    internal_static_tribuo_common_tree_LeafNodeProto_ScoreEntry_descriptor =
      internal_static_tribuo_common_tree_LeafNodeProto_descriptor.getNestedTypes().get(0);
    internal_static_tribuo_common_tree_LeafNodeProto_ScoreEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_common_tree_LeafNodeProto_ScoreEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    internal_static_tribuo_common_tree_TreeModelProto_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_tribuo_common_tree_TreeModelProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tribuo_common_tree_TreeModelProto_descriptor,
        new java.lang.String[] { "Metadata", "Nodes", });
    com.google.protobuf.AnyProto.getDescriptor();
    org.tribuo.protos.core.TribuoCore.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}

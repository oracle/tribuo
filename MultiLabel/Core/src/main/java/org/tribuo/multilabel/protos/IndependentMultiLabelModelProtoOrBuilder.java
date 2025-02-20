// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-multilabel-core.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.multilabel.protos;

public interface IndependentMultiLabelModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.multilabel.IndependentMultiLabelModelProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   * @return Whether the metadata field is set.
   */
  boolean hasMetadata();
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   * @return The metadata.
   */
  org.tribuo.protos.core.ModelDataProto getMetadata();
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   */
  org.tribuo.protos.core.ModelDataProtoOrBuilder getMetadataOrBuilder();

  /**
   * <code>repeated .tribuo.core.ModelProto models = 2;</code>
   */
  java.util.List<org.tribuo.protos.core.ModelProto> 
      getModelsList();
  /**
   * <code>repeated .tribuo.core.ModelProto models = 2;</code>
   */
  org.tribuo.protos.core.ModelProto getModels(int index);
  /**
   * <code>repeated .tribuo.core.ModelProto models = 2;</code>
   */
  int getModelsCount();
  /**
   * <code>repeated .tribuo.core.ModelProto models = 2;</code>
   */
  java.util.List<? extends org.tribuo.protos.core.ModelProtoOrBuilder> 
      getModelsOrBuilderList();
  /**
   * <code>repeated .tribuo.core.ModelProto models = 2;</code>
   */
  org.tribuo.protos.core.ModelProtoOrBuilder getModelsOrBuilder(
      int index);

  /**
   * <code>repeated .tribuo.core.OutputProto labels = 3;</code>
   */
  java.util.List<org.tribuo.protos.core.OutputProto> 
      getLabelsList();
  /**
   * <code>repeated .tribuo.core.OutputProto labels = 3;</code>
   */
  org.tribuo.protos.core.OutputProto getLabels(int index);
  /**
   * <code>repeated .tribuo.core.OutputProto labels = 3;</code>
   */
  int getLabelsCount();
  /**
   * <code>repeated .tribuo.core.OutputProto labels = 3;</code>
   */
  java.util.List<? extends org.tribuo.protos.core.OutputProtoOrBuilder> 
      getLabelsOrBuilderList();
  /**
   * <code>repeated .tribuo.core.OutputProto labels = 3;</code>
   */
  org.tribuo.protos.core.OutputProtoOrBuilder getLabelsOrBuilder(
      int index);
}

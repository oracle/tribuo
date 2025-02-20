// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-xgboost.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.common.xgboost.protos;

public interface XGBoostExternalModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.common.xgboost.XGBoostExternalModelProto)
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
   * <code>.tribuo.common.xgboost.XGBoostOutputConverterProto converter = 2;</code>
   * @return Whether the converter field is set.
   */
  boolean hasConverter();
  /**
   * <code>.tribuo.common.xgboost.XGBoostOutputConverterProto converter = 2;</code>
   * @return The converter.
   */
  org.tribuo.common.xgboost.protos.XGBoostOutputConverterProto getConverter();
  /**
   * <code>.tribuo.common.xgboost.XGBoostOutputConverterProto converter = 2;</code>
   */
  org.tribuo.common.xgboost.protos.XGBoostOutputConverterProtoOrBuilder getConverterOrBuilder();

  /**
   * <code>bytes model = 3;</code>
   * @return The model.
   */
  com.google.protobuf.ByteString getModel();

  /**
   * <code>repeated int32 forward_feature_mapping = 4;</code>
   * @return A list containing the forwardFeatureMapping.
   */
  java.util.List<java.lang.Integer> getForwardFeatureMappingList();
  /**
   * <code>repeated int32 forward_feature_mapping = 4;</code>
   * @return The count of forwardFeatureMapping.
   */
  int getForwardFeatureMappingCount();
  /**
   * <code>repeated int32 forward_feature_mapping = 4;</code>
   * @param index The index of the element to return.
   * @return The forwardFeatureMapping at the given index.
   */
  int getForwardFeatureMapping(int index);

  /**
   * <code>repeated int32 backward_feature_mapping = 5;</code>
   * @return A list containing the backwardFeatureMapping.
   */
  java.util.List<java.lang.Integer> getBackwardFeatureMappingList();
  /**
   * <code>repeated int32 backward_feature_mapping = 5;</code>
   * @return The count of backwardFeatureMapping.
   */
  int getBackwardFeatureMappingCount();
  /**
   * <code>repeated int32 backward_feature_mapping = 5;</code>
   * @param index The index of the element to return.
   * @return The backwardFeatureMapping at the given index.
   */
  int getBackwardFeatureMapping(int index);
}

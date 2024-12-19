// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.protos.core;

public interface SelectedFeatureDatasetProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.SelectedFeatureDatasetProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.tribuo.core.DatasetDataProto metadata = 1;</code>
   * @return Whether the metadata field is set.
   */
  boolean hasMetadata();
  /**
   * <code>.tribuo.core.DatasetDataProto metadata = 1;</code>
   * @return The metadata.
   */
  org.tribuo.protos.core.DatasetDataProto getMetadata();
  /**
   * <code>.tribuo.core.DatasetDataProto metadata = 1;</code>
   */
  org.tribuo.protos.core.DatasetDataProtoOrBuilder getMetadataOrBuilder();

  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 2;</code>
   */
  java.util.List<org.tribuo.protos.core.ExampleProto> 
      getExamplesList();
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 2;</code>
   */
  org.tribuo.protos.core.ExampleProto getExamples(int index);
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 2;</code>
   */
  int getExamplesCount();
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 2;</code>
   */
  java.util.List<? extends org.tribuo.protos.core.ExampleProtoOrBuilder> 
      getExamplesOrBuilderList();
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 2;</code>
   */
  org.tribuo.protos.core.ExampleProtoOrBuilder getExamplesOrBuilder(
      int index);

  /**
   * <code>int32 k = 3;</code>
   * @return The k.
   */
  int getK();

  /**
   * <code>.tribuo.core.FeatureSetProto feature_set = 4;</code>
   * @return Whether the featureSet field is set.
   */
  boolean hasFeatureSet();
  /**
   * <code>.tribuo.core.FeatureSetProto feature_set = 4;</code>
   * @return The featureSet.
   */
  org.tribuo.protos.core.FeatureSetProto getFeatureSet();
  /**
   * <code>.tribuo.core.FeatureSetProto feature_set = 4;</code>
   */
  org.tribuo.protos.core.FeatureSetProtoOrBuilder getFeatureSetOrBuilder();

  /**
   * <code>repeated string selected_features = 5;</code>
   * @return A list containing the selectedFeatures.
   */
  java.util.List<java.lang.String>
      getSelectedFeaturesList();
  /**
   * <code>repeated string selected_features = 5;</code>
   * @return The count of selectedFeatures.
   */
  int getSelectedFeaturesCount();
  /**
   * <code>repeated string selected_features = 5;</code>
   * @param index The index of the element to return.
   * @return The selectedFeatures at the given index.
   */
  java.lang.String getSelectedFeatures(int index);
  /**
   * <code>repeated string selected_features = 5;</code>
   * @param index The index of the value to return.
   * @return The bytes of the selectedFeatures at the given index.
   */
  com.google.protobuf.ByteString
      getSelectedFeaturesBytes(int index);

  /**
   * <code>int32 num_examples_removed = 6;</code>
   * @return The numExamplesRemoved.
   */
  int getNumExamplesRemoved();
}

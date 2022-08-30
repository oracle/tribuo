// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

package org.tribuo.protos.core;

public interface MinimumCardinalityDatasetProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.MinimumCardinalityDatasetProto)
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
   * <code>bool drop_invalid_examples = 3;</code>
   * @return The dropInvalidExamples.
   */
  boolean getDropInvalidExamples();

  /**
   * <code>int32 min_cardinality = 4;</code>
   * @return The minCardinality.
   */
  int getMinCardinality();

  /**
   * <code>int32 num_examples_removed = 5;</code>
   * @return The numExamplesRemoved.
   */
  int getNumExamplesRemoved();

  /**
   * <code>repeated string removed = 6;</code>
   * @return A list containing the removed.
   */
  java.util.List<java.lang.String>
      getRemovedList();
  /**
   * <code>repeated string removed = 6;</code>
   * @return The count of removed.
   */
  int getRemovedCount();
  /**
   * <code>repeated string removed = 6;</code>
   * @param index The index of the element to return.
   * @return The removed at the given index.
   */
  java.lang.String getRemoved(int index);
  /**
   * <code>repeated string removed = 6;</code>
   * @param index The index of the value to return.
   * @return The bytes of the removed at the given index.
   */
  com.google.protobuf.ByteString
      getRemovedBytes(int index);
}

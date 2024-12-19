// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.protos.core;

public interface ImmutableDatasetProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.ImmutableDatasetProto)
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
}

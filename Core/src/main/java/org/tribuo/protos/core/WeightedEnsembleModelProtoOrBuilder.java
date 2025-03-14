// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.protos.core;

public interface WeightedEnsembleModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.WeightedEnsembleModelProto)
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
   * <code>repeated float weights = 3;</code>
   * @return A list containing the weights.
   */
  java.util.List<java.lang.Float> getWeightsList();
  /**
   * <code>repeated float weights = 3;</code>
   * @return The count of weights.
   */
  int getWeightsCount();
  /**
   * <code>repeated float weights = 3;</code>
   * @param index The index of the element to return.
   * @return The weights at the given index.
   */
  float getWeights(int index);

  /**
   * <code>.tribuo.core.EnsembleCombinerProto combiner = 4;</code>
   * @return Whether the combiner field is set.
   */
  boolean hasCombiner();
  /**
   * <code>.tribuo.core.EnsembleCombinerProto combiner = 4;</code>
   * @return The combiner.
   */
  org.tribuo.protos.core.EnsembleCombinerProto getCombiner();
  /**
   * <code>.tribuo.core.EnsembleCombinerProto combiner = 4;</code>
   */
  org.tribuo.protos.core.EnsembleCombinerProtoOrBuilder getCombinerOrBuilder();
}

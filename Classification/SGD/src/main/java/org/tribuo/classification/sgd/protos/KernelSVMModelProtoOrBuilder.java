// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-classification-sgd.proto

package org.tribuo.classification.sgd.protos;

public interface KernelSVMModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.classification.sgd.KernelSVMModelProto)
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
   * <code>.tribuo.math.KernelProto kernel = 2;</code>
   * @return Whether the kernel field is set.
   */
  boolean hasKernel();
  /**
   * <code>.tribuo.math.KernelProto kernel = 2;</code>
   * @return The kernel.
   */
  org.tribuo.math.protos.KernelProto getKernel();
  /**
   * <code>.tribuo.math.KernelProto kernel = 2;</code>
   */
  org.tribuo.math.protos.KernelProtoOrBuilder getKernelOrBuilder();

  /**
   * <code>.tribuo.math.TensorProto weights = 3;</code>
   * @return Whether the weights field is set.
   */
  boolean hasWeights();
  /**
   * <code>.tribuo.math.TensorProto weights = 3;</code>
   * @return The weights.
   */
  org.tribuo.math.protos.TensorProto getWeights();
  /**
   * <code>.tribuo.math.TensorProto weights = 3;</code>
   */
  org.tribuo.math.protos.TensorProtoOrBuilder getWeightsOrBuilder();

  /**
   * <code>repeated .tribuo.math.TensorProto support_vectors = 4;</code>
   */
  java.util.List<org.tribuo.math.protos.TensorProto> 
      getSupportVectorsList();
  /**
   * <code>repeated .tribuo.math.TensorProto support_vectors = 4;</code>
   */
  org.tribuo.math.protos.TensorProto getSupportVectors(int index);
  /**
   * <code>repeated .tribuo.math.TensorProto support_vectors = 4;</code>
   */
  int getSupportVectorsCount();
  /**
   * <code>repeated .tribuo.math.TensorProto support_vectors = 4;</code>
   */
  java.util.List<? extends org.tribuo.math.protos.TensorProtoOrBuilder> 
      getSupportVectorsOrBuilderList();
  /**
   * <code>repeated .tribuo.math.TensorProto support_vectors = 4;</code>
   */
  org.tribuo.math.protos.TensorProtoOrBuilder getSupportVectorsOrBuilder(
      int index);
}

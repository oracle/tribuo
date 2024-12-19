// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-regression-libsvm.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.regression.libsvm.protos;

public interface LibSVMRegressionModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.regression.libsvm.LibSVMRegressionModelProto)
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
   * <code>repeated .tribuo.common.libsvm.SVMModelProto model = 2;</code>
   */
  java.util.List<org.tribuo.common.libsvm.protos.SVMModelProto> 
      getModelList();
  /**
   * <code>repeated .tribuo.common.libsvm.SVMModelProto model = 2;</code>
   */
  org.tribuo.common.libsvm.protos.SVMModelProto getModel(int index);
  /**
   * <code>repeated .tribuo.common.libsvm.SVMModelProto model = 2;</code>
   */
  int getModelCount();
  /**
   * <code>repeated .tribuo.common.libsvm.SVMModelProto model = 2;</code>
   */
  java.util.List<? extends org.tribuo.common.libsvm.protos.SVMModelProtoOrBuilder> 
      getModelOrBuilderList();
  /**
   * <code>repeated .tribuo.common.libsvm.SVMModelProto model = 2;</code>
   */
  org.tribuo.common.libsvm.protos.SVMModelProtoOrBuilder getModelOrBuilder(
      int index);

  /**
   * <code>repeated double means = 4;</code>
   * @return A list containing the means.
   */
  java.util.List<java.lang.Double> getMeansList();
  /**
   * <code>repeated double means = 4;</code>
   * @return The count of means.
   */
  int getMeansCount();
  /**
   * <code>repeated double means = 4;</code>
   * @param index The index of the element to return.
   * @return The means at the given index.
   */
  double getMeans(int index);

  /**
   * <code>repeated double variances = 5;</code>
   * @return A list containing the variances.
   */
  java.util.List<java.lang.Double> getVariancesList();
  /**
   * <code>repeated double variances = 5;</code>
   * @return The count of variances.
   */
  int getVariancesCount();
  /**
   * <code>repeated double variances = 5;</code>
   * @param index The index of the element to return.
   * @return The variances at the given index.
   */
  double getVariances(int index);

  /**
   * <code>bool standardized = 6;</code>
   * @return The standardized.
   */
  boolean getStandardized();
}

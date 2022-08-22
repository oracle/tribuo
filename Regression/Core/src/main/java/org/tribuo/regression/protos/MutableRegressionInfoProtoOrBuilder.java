// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-regression-core.proto

package org.tribuo.regression.protos;

public interface MutableRegressionInfoProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.regression.MutableRegressionInfoProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated string label = 1;</code>
   * @return A list containing the label.
   */
  java.util.List<java.lang.String>
      getLabelList();
  /**
   * <code>repeated string label = 1;</code>
   * @return The count of label.
   */
  int getLabelCount();
  /**
   * <code>repeated string label = 1;</code>
   * @param index The index of the element to return.
   * @return The label at the given index.
   */
  java.lang.String getLabel(int index);
  /**
   * <code>repeated string label = 1;</code>
   * @param index The index of the value to return.
   * @return The bytes of the label at the given index.
   */
  com.google.protobuf.ByteString
      getLabelBytes(int index);

  /**
   * <code>repeated int64 count = 2;</code>
   * @return A list containing the count.
   */
  java.util.List<java.lang.Long> getCountList();
  /**
   * <code>repeated int64 count = 2;</code>
   * @return The count of count.
   */
  int getCountCount();
  /**
   * <code>repeated int64 count = 2;</code>
   * @param index The index of the element to return.
   * @return The count at the given index.
   */
  long getCount(int index);

  /**
   * <code>repeated double max = 3;</code>
   * @return A list containing the max.
   */
  java.util.List<java.lang.Double> getMaxList();
  /**
   * <code>repeated double max = 3;</code>
   * @return The count of max.
   */
  int getMaxCount();
  /**
   * <code>repeated double max = 3;</code>
   * @param index The index of the element to return.
   * @return The max at the given index.
   */
  double getMax(int index);

  /**
   * <code>repeated double min = 4;</code>
   * @return A list containing the min.
   */
  java.util.List<java.lang.Double> getMinList();
  /**
   * <code>repeated double min = 4;</code>
   * @return The count of min.
   */
  int getMinCount();
  /**
   * <code>repeated double min = 4;</code>
   * @param index The index of the element to return.
   * @return The min at the given index.
   */
  double getMin(int index);

  /**
   * <code>repeated double mean = 5;</code>
   * @return A list containing the mean.
   */
  java.util.List<java.lang.Double> getMeanList();
  /**
   * <code>repeated double mean = 5;</code>
   * @return The count of mean.
   */
  int getMeanCount();
  /**
   * <code>repeated double mean = 5;</code>
   * @param index The index of the element to return.
   * @return The mean at the given index.
   */
  double getMean(int index);

  /**
   * <code>repeated double sumSquares = 6;</code>
   * @return A list containing the sumSquares.
   */
  java.util.List<java.lang.Double> getSumSquaresList();
  /**
   * <code>repeated double sumSquares = 6;</code>
   * @return The count of sumSquares.
   */
  int getSumSquaresCount();
  /**
   * <code>repeated double sumSquares = 6;</code>
   * @param index The index of the element to return.
   * @return The sumSquares at the given index.
   */
  double getSumSquares(int index);

  /**
   * <code>int32 unknownCount = 7;</code>
   * @return The unknownCount.
   */
  int getUnknownCount();

  /**
   * <code>int64 overallCount = 8;</code>
   * @return The overallCount.
   */
  long getOverallCount();
}

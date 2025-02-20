// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.protos.core;

public interface BinningTransformerProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.BinningTransformerProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>string binning_type = 1;</code>
   * @return The binningType.
   */
  java.lang.String getBinningType();
  /**
   * <code>string binning_type = 1;</code>
   * @return The bytes for binningType.
   */
  com.google.protobuf.ByteString
      getBinningTypeBytes();

  /**
   * <code>repeated double bins = 2;</code>
   * @return A list containing the bins.
   */
  java.util.List<java.lang.Double> getBinsList();
  /**
   * <code>repeated double bins = 2;</code>
   * @return The count of bins.
   */
  int getBinsCount();
  /**
   * <code>repeated double bins = 2;</code>
   * @param index The index of the element to return.
   * @return The bins at the given index.
   */
  double getBins(int index);

  /**
   * <code>repeated double values = 3;</code>
   * @return A list containing the values.
   */
  java.util.List<java.lang.Double> getValuesList();
  /**
   * <code>repeated double values = 3;</code>
   * @return The count of values.
   */
  int getValuesCount();
  /**
   * <code>repeated double values = 3;</code>
   * @param index The index of the element to return.
   * @return The values at the given index.
   */
  double getValues(int index);
}

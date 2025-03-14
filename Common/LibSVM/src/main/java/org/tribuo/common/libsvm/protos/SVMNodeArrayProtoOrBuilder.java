// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-libsvm.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.common.libsvm.protos;

public interface SVMNodeArrayProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.common.libsvm.SVMNodeArrayProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated int32 index = 1;</code>
   * @return A list containing the index.
   */
  java.util.List<java.lang.Integer> getIndexList();
  /**
   * <code>repeated int32 index = 1;</code>
   * @return The count of index.
   */
  int getIndexCount();
  /**
   * <code>repeated int32 index = 1;</code>
   * @param index The index of the element to return.
   * @return The index at the given index.
   */
  int getIndex(int index);

  /**
   * <code>repeated double value = 2;</code>
   * @return A list containing the value.
   */
  java.util.List<java.lang.Double> getValueList();
  /**
   * <code>repeated double value = 2;</code>
   * @return The count of value.
   */
  int getValueCount();
  /**
   * <code>repeated double value = 2;</code>
   * @param index The index of the element to return.
   * @return The value at the given index.
   */
  double getValue(int index);
}

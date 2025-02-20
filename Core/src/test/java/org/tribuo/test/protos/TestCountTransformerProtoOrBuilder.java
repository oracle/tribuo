// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-test.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.test.protos;

public interface TestCountTransformerProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.TestCountTransformerProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>int32 count = 1;</code>
   * @return The count.
   */
  int getCount();

  /**
   * <code>int32 sparseCount = 2;</code>
   * @return The sparseCount.
   */
  int getSparseCount();

  /**
   * <code>repeated double countMapKeys = 3;</code>
   * @return A list containing the countMapKeys.
   */
  java.util.List<java.lang.Double> getCountMapKeysList();
  /**
   * <code>repeated double countMapKeys = 3;</code>
   * @return The count of countMapKeys.
   */
  int getCountMapKeysCount();
  /**
   * <code>repeated double countMapKeys = 3;</code>
   * @param index The index of the element to return.
   * @return The countMapKeys at the given index.
   */
  double getCountMapKeys(int index);

  /**
   * <code>repeated int64 countMapValues = 4;</code>
   * @return A list containing the countMapValues.
   */
  java.util.List<java.lang.Long> getCountMapValuesList();
  /**
   * <code>repeated int64 countMapValues = 4;</code>
   * @return The count of countMapValues.
   */
  int getCountMapValuesCount();
  /**
   * <code>repeated int64 countMapValues = 4;</code>
   * @param index The index of the element to return.
   * @return The countMapValues at the given index.
   */
  long getCountMapValues(int index);
}

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-oci.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.interop.oci.protos;

public interface OCIOutputConverterProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.interop.oci.OCIOutputConverterProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>int32 version = 1;</code>
   * @return The version.
   */
  int getVersion();

  /**
   * <code>string class_name = 2;</code>
   * @return The className.
   */
  java.lang.String getClassName();
  /**
   * <code>string class_name = 2;</code>
   * @return The bytes for className.
   */
  com.google.protobuf.ByteString
      getClassNameBytes();

  /**
   * <code>.google.protobuf.Any serialized_data = 3;</code>
   * @return Whether the serializedData field is set.
   */
  boolean hasSerializedData();
  /**
   * <code>.google.protobuf.Any serialized_data = 3;</code>
   * @return The serializedData.
   */
  com.google.protobuf.Any getSerializedData();
  /**
   * <code>.google.protobuf.Any serialized_data = 3;</code>
   */
  com.google.protobuf.AnyOrBuilder getSerializedDataOrBuilder();
}

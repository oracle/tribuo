// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-tensorflow.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.interop.tensorflow.protos;

public interface ImageConverterProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.interop.tensorflow.ImageConverterProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>string input_name = 1;</code>
   * @return The inputName.
   */
  java.lang.String getInputName();
  /**
   * <code>string input_name = 1;</code>
   * @return The bytes for inputName.
   */
  com.google.protobuf.ByteString
      getInputNameBytes();

  /**
   * <code>int32 width = 2;</code>
   * @return The width.
   */
  int getWidth();

  /**
   * <code>int32 height = 3;</code>
   * @return The height.
   */
  int getHeight();

  /**
   * <code>int32 channels = 4;</code>
   * @return The channels.
   */
  int getChannels();
}

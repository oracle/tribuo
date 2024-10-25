// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-liblinear.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.common.liblinear.protos;

public interface LibLinearProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.common.liblinear.LibLinearProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>double bias = 1;</code>
   * @return The bias.
   */
  double getBias();

  /**
   * <code>repeated int32 label = 2;</code>
   * @return A list containing the label.
   */
  java.util.List<java.lang.Integer> getLabelList();
  /**
   * <code>repeated int32 label = 2;</code>
   * @return The count of label.
   */
  int getLabelCount();
  /**
   * <code>repeated int32 label = 2;</code>
   * @param index The index of the element to return.
   * @return The label at the given index.
   */
  int getLabel(int index);

  /**
   * <code>int32 nr_class = 3;</code>
   * @return The nrClass.
   */
  int getNrClass();

  /**
   * <code>int32 nr_feature = 4;</code>
   * @return The nrFeature.
   */
  int getNrFeature();

  /**
   * <code>string solver_type = 5;</code>
   * @return The solverType.
   */
  java.lang.String getSolverType();
  /**
   * <code>string solver_type = 5;</code>
   * @return The bytes for solverType.
   */
  com.google.protobuf.ByteString
      getSolverTypeBytes();

  /**
   * <code>repeated double w = 6;</code>
   * @return A list containing the w.
   */
  java.util.List<java.lang.Double> getWList();
  /**
   * <code>repeated double w = 6;</code>
   * @return The count of w.
   */
  int getWCount();
  /**
   * <code>repeated double w = 6;</code>
   * @param index The index of the element to return.
   * @return The w at the given index.
   */
  double getW(int index);

  /**
   * <code>double rho = 7;</code>
   * @return The rho.
   */
  double getRho();
}

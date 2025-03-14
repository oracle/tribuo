// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-libsvm.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.common.libsvm.protos;

public interface SVMModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.common.libsvm.SVMModelProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.tribuo.common.libsvm.SVMParameterProto param = 1;</code>
   * @return Whether the param field is set.
   */
  boolean hasParam();
  /**
   * <code>.tribuo.common.libsvm.SVMParameterProto param = 1;</code>
   * @return The param.
   */
  org.tribuo.common.libsvm.protos.SVMParameterProto getParam();
  /**
   * <code>.tribuo.common.libsvm.SVMParameterProto param = 1;</code>
   */
  org.tribuo.common.libsvm.protos.SVMParameterProtoOrBuilder getParamOrBuilder();

  /**
   * <code>int32 nr_class = 2;</code>
   * @return The nrClass.
   */
  int getNrClass();

  /**
   * <code>int32 l = 3;</code>
   * @return The l.
   */
  int getL();

  /**
   * <code>repeated .tribuo.common.libsvm.SVMNodeArrayProto SV = 4;</code>
   */
  java.util.List<org.tribuo.common.libsvm.protos.SVMNodeArrayProto> 
      getSVList();
  /**
   * <code>repeated .tribuo.common.libsvm.SVMNodeArrayProto SV = 4;</code>
   */
  org.tribuo.common.libsvm.protos.SVMNodeArrayProto getSV(int index);
  /**
   * <code>repeated .tribuo.common.libsvm.SVMNodeArrayProto SV = 4;</code>
   */
  int getSVCount();
  /**
   * <code>repeated .tribuo.common.libsvm.SVMNodeArrayProto SV = 4;</code>
   */
  java.util.List<? extends org.tribuo.common.libsvm.protos.SVMNodeArrayProtoOrBuilder> 
      getSVOrBuilderList();
  /**
   * <code>repeated .tribuo.common.libsvm.SVMNodeArrayProto SV = 4;</code>
   */
  org.tribuo.common.libsvm.protos.SVMNodeArrayProtoOrBuilder getSVOrBuilder(
      int index);

  /**
   * <code>repeated int32 sv_coef_lengths = 5;</code>
   * @return A list containing the svCoefLengths.
   */
  java.util.List<java.lang.Integer> getSvCoefLengthsList();
  /**
   * <code>repeated int32 sv_coef_lengths = 5;</code>
   * @return The count of svCoefLengths.
   */
  int getSvCoefLengthsCount();
  /**
   * <code>repeated int32 sv_coef_lengths = 5;</code>
   * @param index The index of the element to return.
   * @return The svCoefLengths at the given index.
   */
  int getSvCoefLengths(int index);

  /**
   * <code>repeated double sv_coef = 6;</code>
   * @return A list containing the svCoef.
   */
  java.util.List<java.lang.Double> getSvCoefList();
  /**
   * <code>repeated double sv_coef = 6;</code>
   * @return The count of svCoef.
   */
  int getSvCoefCount();
  /**
   * <code>repeated double sv_coef = 6;</code>
   * @param index The index of the element to return.
   * @return The svCoef at the given index.
   */
  double getSvCoef(int index);

  /**
   * <code>repeated double rho = 7;</code>
   * @return A list containing the rho.
   */
  java.util.List<java.lang.Double> getRhoList();
  /**
   * <code>repeated double rho = 7;</code>
   * @return The count of rho.
   */
  int getRhoCount();
  /**
   * <code>repeated double rho = 7;</code>
   * @param index The index of the element to return.
   * @return The rho at the given index.
   */
  double getRho(int index);

  /**
   * <code>repeated double probA = 8;</code>
   * @return A list containing the probA.
   */
  java.util.List<java.lang.Double> getProbAList();
  /**
   * <code>repeated double probA = 8;</code>
   * @return The count of probA.
   */
  int getProbACount();
  /**
   * <code>repeated double probA = 8;</code>
   * @param index The index of the element to return.
   * @return The probA at the given index.
   */
  double getProbA(int index);

  /**
   * <code>repeated double probB = 9;</code>
   * @return A list containing the probB.
   */
  java.util.List<java.lang.Double> getProbBList();
  /**
   * <code>repeated double probB = 9;</code>
   * @return The count of probB.
   */
  int getProbBCount();
  /**
   * <code>repeated double probB = 9;</code>
   * @param index The index of the element to return.
   * @return The probB at the given index.
   */
  double getProbB(int index);

  /**
   * <code>repeated int32 sv_indices = 10;</code>
   * @return A list containing the svIndices.
   */
  java.util.List<java.lang.Integer> getSvIndicesList();
  /**
   * <code>repeated int32 sv_indices = 10;</code>
   * @return The count of svIndices.
   */
  int getSvIndicesCount();
  /**
   * <code>repeated int32 sv_indices = 10;</code>
   * @param index The index of the element to return.
   * @return The svIndices at the given index.
   */
  int getSvIndices(int index);

  /**
   * <code>repeated int32 label = 11;</code>
   * @return A list containing the label.
   */
  java.util.List<java.lang.Integer> getLabelList();
  /**
   * <code>repeated int32 label = 11;</code>
   * @return The count of label.
   */
  int getLabelCount();
  /**
   * <code>repeated int32 label = 11;</code>
   * @param index The index of the element to return.
   * @return The label at the given index.
   */
  int getLabel(int index);

  /**
   * <code>repeated int32 nSV = 12;</code>
   * @return A list containing the nSV.
   */
  java.util.List<java.lang.Integer> getNSVList();
  /**
   * <code>repeated int32 nSV = 12;</code>
   * @return The count of nSV.
   */
  int getNSVCount();
  /**
   * <code>repeated int32 nSV = 12;</code>
   * @param index The index of the element to return.
   * @return The nSV at the given index.
   */
  int getNSV(int index);
}

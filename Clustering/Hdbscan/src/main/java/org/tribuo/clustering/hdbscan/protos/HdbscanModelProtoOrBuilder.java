// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-clustering-hdbscan.proto

package org.tribuo.clustering.hdbscan.protos;

public interface HdbscanModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.clustering.hdbscan.HdbscanModelProto)
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
   * <code>repeated int32 cluster_labels = 2;</code>
   * @return A list containing the clusterLabels.
   */
  java.util.List<java.lang.Integer> getClusterLabelsList();
  /**
   * <code>repeated int32 cluster_labels = 2;</code>
   * @return The count of clusterLabels.
   */
  int getClusterLabelsCount();
  /**
   * <code>repeated int32 cluster_labels = 2;</code>
   * @param index The index of the element to return.
   * @return The clusterLabels at the given index.
   */
  int getClusterLabels(int index);

  /**
   * <code>.tribuo.math.TensorProto outlier_scores_vector = 3;</code>
   * @return Whether the outlierScoresVector field is set.
   */
  boolean hasOutlierScoresVector();
  /**
   * <code>.tribuo.math.TensorProto outlier_scores_vector = 3;</code>
   * @return The outlierScoresVector.
   */
  org.tribuo.math.protos.TensorProto getOutlierScoresVector();
  /**
   * <code>.tribuo.math.TensorProto outlier_scores_vector = 3;</code>
   */
  org.tribuo.math.protos.TensorProtoOrBuilder getOutlierScoresVectorOrBuilder();

  /**
   * <code>string dist_type = 4;</code>
   * @return The distType.
   */
  java.lang.String getDistType();
  /**
   * <code>string dist_type = 4;</code>
   * @return The bytes for distType.
   */
  com.google.protobuf.ByteString
      getDistTypeBytes();

  /**
   * <code>repeated .tribuo.clustering.hdbscan.ClusterExemplarProto cluster_exemplars = 5;</code>
   */
  java.util.List<org.tribuo.clustering.hdbscan.protos.ClusterExemplarProto> 
      getClusterExemplarsList();
  /**
   * <code>repeated .tribuo.clustering.hdbscan.ClusterExemplarProto cluster_exemplars = 5;</code>
   */
  org.tribuo.clustering.hdbscan.protos.ClusterExemplarProto getClusterExemplars(int index);
  /**
   * <code>repeated .tribuo.clustering.hdbscan.ClusterExemplarProto cluster_exemplars = 5;</code>
   */
  int getClusterExemplarsCount();
  /**
   * <code>repeated .tribuo.clustering.hdbscan.ClusterExemplarProto cluster_exemplars = 5;</code>
   */
  java.util.List<? extends org.tribuo.clustering.hdbscan.protos.ClusterExemplarProtoOrBuilder> 
      getClusterExemplarsOrBuilderList();
  /**
   * <code>repeated .tribuo.clustering.hdbscan.ClusterExemplarProto cluster_exemplars = 5;</code>
   */
  org.tribuo.clustering.hdbscan.protos.ClusterExemplarProtoOrBuilder getClusterExemplarsOrBuilder(
      int index);

  /**
   * <code>double noise_points_outlier_score = 6;</code>
   * @return The noisePointsOutlierScore.
   */
  double getNoisePointsOutlierScore();
}

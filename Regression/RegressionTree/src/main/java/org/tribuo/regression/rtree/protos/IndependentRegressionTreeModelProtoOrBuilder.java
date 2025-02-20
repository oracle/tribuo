// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-regression-tree.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.regression.rtree.protos;

public interface IndependentRegressionTreeModelProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.regression.tree.IndependentRegressionTreeModelProto)
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
   * <code>map&lt;string, .tribuo.regression.tree.TreeNodeListProto&gt; nodes = 2;</code>
   */
  int getNodesCount();
  /**
   * <code>map&lt;string, .tribuo.regression.tree.TreeNodeListProto&gt; nodes = 2;</code>
   */
  boolean containsNodes(
      java.lang.String key);
  /**
   * Use {@link #getNodesMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.tribuo.regression.rtree.protos.TreeNodeListProto>
  getNodes();
  /**
   * <code>map&lt;string, .tribuo.regression.tree.TreeNodeListProto&gt; nodes = 2;</code>
   */
  java.util.Map<java.lang.String, org.tribuo.regression.rtree.protos.TreeNodeListProto>
  getNodesMap();
  /**
   * <code>map&lt;string, .tribuo.regression.tree.TreeNodeListProto&gt; nodes = 2;</code>
   */
  /* nullable */
org.tribuo.regression.rtree.protos.TreeNodeListProto getNodesOrDefault(
      java.lang.String key,
      /* nullable */
org.tribuo.regression.rtree.protos.TreeNodeListProto defaultValue);
  /**
   * <code>map&lt;string, .tribuo.regression.tree.TreeNodeListProto&gt; nodes = 2;</code>
   */
  org.tribuo.regression.rtree.protos.TreeNodeListProto getNodesOrThrow(
      java.lang.String key);
}

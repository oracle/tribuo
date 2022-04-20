// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core.proto

package org.tribuo.protos.core;

public interface TransformerMapProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.TransformerMapProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>int32 version = 1;</code>
   * @return The version.
   */
  int getVersion();

  /**
   * <code>map&lt;string, .tribuo.core.TransformerListProto&gt; transformers = 2;</code>
   */
  int getTransformersCount();
  /**
   * <code>map&lt;string, .tribuo.core.TransformerListProto&gt; transformers = 2;</code>
   */
  boolean containsTransformers(
      java.lang.String key);
  /**
   * Use {@link #getTransformersMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.tribuo.protos.core.TransformerListProto>
  getTransformers();
  /**
   * <code>map&lt;string, .tribuo.core.TransformerListProto&gt; transformers = 2;</code>
   */
  java.util.Map<java.lang.String, org.tribuo.protos.core.TransformerListProto>
  getTransformersMap();
  /**
   * <code>map&lt;string, .tribuo.core.TransformerListProto&gt; transformers = 2;</code>
   */

  /* nullable */
org.tribuo.protos.core.TransformerListProto getTransformersOrDefault(
      java.lang.String key,
      /* nullable */
org.tribuo.protos.core.TransformerListProto defaultValue);
  /**
   * <code>map&lt;string, .tribuo.core.TransformerListProto&gt; transformers = 2;</code>
   */

  org.tribuo.protos.core.TransformerListProto getTransformersOrThrow(
      java.lang.String key);

  /**
   * <code>.olcut.RootProvenanceProto datasetProvenance = 3;</code>
   * @return Whether the datasetProvenance field is set.
   */
  boolean hasDatasetProvenance();
  /**
   * <code>.olcut.RootProvenanceProto datasetProvenance = 3;</code>
   * @return The datasetProvenance.
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.RootProvenanceProto getDatasetProvenance();
  /**
   * <code>.olcut.RootProvenanceProto datasetProvenance = 3;</code>
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.RootProvenanceProtoOrBuilder getDatasetProvenanceOrBuilder();

  /**
   * <code>.olcut.RootProvenanceProto transformationMapProvenance = 4;</code>
   * @return Whether the transformationMapProvenance field is set.
   */
  boolean hasTransformationMapProvenance();
  /**
   * <code>.olcut.RootProvenanceProto transformationMapProvenance = 4;</code>
   * @return The transformationMapProvenance.
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.RootProvenanceProto getTransformationMapProvenance();
  /**
   * <code>.olcut.RootProvenanceProto transformationMapProvenance = 4;</code>
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.RootProvenanceProtoOrBuilder getTransformationMapProvenanceOrBuilder();
}

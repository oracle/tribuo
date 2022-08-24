// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core.proto

package org.tribuo.protos.core;

public interface DatasetDataProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.DatasetDataProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.olcut.RootProvenanceProto provenance = 3;</code>
   * @return Whether the provenance field is set.
   */
  boolean hasProvenance();
  /**
   * <code>.olcut.RootProvenanceProto provenance = 3;</code>
   * @return The provenance.
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.RootProvenanceProto getProvenance();
  /**
   * <code>.olcut.RootProvenanceProto provenance = 3;</code>
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.RootProvenanceProtoOrBuilder getProvenanceOrBuilder();

  /**
   * <code>.tribuo.core.FeatureDomainProto feature_domain = 4;</code>
   * @return Whether the featureDomain field is set.
   */
  boolean hasFeatureDomain();
  /**
   * <code>.tribuo.core.FeatureDomainProto feature_domain = 4;</code>
   * @return The featureDomain.
   */
  org.tribuo.protos.core.FeatureDomainProto getFeatureDomain();
  /**
   * <code>.tribuo.core.FeatureDomainProto feature_domain = 4;</code>
   */
  org.tribuo.protos.core.FeatureDomainProtoOrBuilder getFeatureDomainOrBuilder();

  /**
   * <code>.tribuo.core.OutputDomainProto output_domain = 5;</code>
   * @return Whether the outputDomain field is set.
   */
  boolean hasOutputDomain();
  /**
   * <code>.tribuo.core.OutputDomainProto output_domain = 5;</code>
   * @return The outputDomain.
   */
  org.tribuo.protos.core.OutputDomainProto getOutputDomain();
  /**
   * <code>.tribuo.core.OutputDomainProto output_domain = 5;</code>
   */
  org.tribuo.protos.core.OutputDomainProtoOrBuilder getOutputDomainOrBuilder();

  /**
   * <code>.olcut.ListProvenanceProto transform_provenance = 7;</code>
   * @return Whether the transformProvenance field is set.
   */
  boolean hasTransformProvenance();
  /**
   * <code>.olcut.ListProvenanceProto transform_provenance = 7;</code>
   * @return The transformProvenance.
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.ListProvenanceProto getTransformProvenance();
  /**
   * <code>.olcut.ListProvenanceProto transform_provenance = 7;</code>
   */
  com.oracle.labs.mlrg.olcut.config.protobuf.protos.ListProvenanceProtoOrBuilder getTransformProvenanceOrBuilder();
}
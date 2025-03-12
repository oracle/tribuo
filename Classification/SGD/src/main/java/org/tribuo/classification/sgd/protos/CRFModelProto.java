// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-classification-sgd.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.classification.sgd.protos;

/**
 * <pre>
 *
 *CRFModel proto
 * </pre>
 *
 * Protobuf type {@code tribuo.classification.sgd.CRFModelProto}
 */
public final class CRFModelProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.classification.sgd.CRFModelProto)
    CRFModelProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use CRFModelProto.newBuilder() to construct.
  private CRFModelProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private CRFModelProto() {
    confidenceType_ = "";
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new CRFModelProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_CRFModelProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_CRFModelProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.classification.sgd.protos.CRFModelProto.class, org.tribuo.classification.sgd.protos.CRFModelProto.Builder.class);
  }

  private int bitField0_;
  public static final int METADATA_FIELD_NUMBER = 1;
  private org.tribuo.protos.core.ModelDataProto metadata_;
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   * @return Whether the metadata field is set.
   */
  @java.lang.Override
  public boolean hasMetadata() {
    return ((bitField0_ & 0x00000001) != 0);
  }
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   * @return The metadata.
   */
  @java.lang.Override
  public org.tribuo.protos.core.ModelDataProto getMetadata() {
    return metadata_ == null ? org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
  }
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   */
  @java.lang.Override
  public org.tribuo.protos.core.ModelDataProtoOrBuilder getMetadataOrBuilder() {
    return metadata_ == null ? org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
  }

  public static final int PARAMS_FIELD_NUMBER = 2;
  private org.tribuo.math.protos.ParametersProto params_;
  /**
   * <code>.tribuo.math.ParametersProto params = 2;</code>
   * @return Whether the params field is set.
   */
  @java.lang.Override
  public boolean hasParams() {
    return ((bitField0_ & 0x00000002) != 0);
  }
  /**
   * <code>.tribuo.math.ParametersProto params = 2;</code>
   * @return The params.
   */
  @java.lang.Override
  public org.tribuo.math.protos.ParametersProto getParams() {
    return params_ == null ? org.tribuo.math.protos.ParametersProto.getDefaultInstance() : params_;
  }
  /**
   * <code>.tribuo.math.ParametersProto params = 2;</code>
   */
  @java.lang.Override
  public org.tribuo.math.protos.ParametersProtoOrBuilder getParamsOrBuilder() {
    return params_ == null ? org.tribuo.math.protos.ParametersProto.getDefaultInstance() : params_;
  }

  public static final int CONFIDENCE_TYPE_FIELD_NUMBER = 3;
  @SuppressWarnings("serial")
  private volatile java.lang.Object confidenceType_ = "";
  /**
   * <code>string confidence_type = 3;</code>
   * @return The confidenceType.
   */
  @java.lang.Override
  public java.lang.String getConfidenceType() {
    java.lang.Object ref = confidenceType_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      confidenceType_ = s;
      return s;
    }
  }
  /**
   * <code>string confidence_type = 3;</code>
   * @return The bytes for confidenceType.
   */
  @java.lang.Override
  public com.google.protobuf.ByteString
      getConfidenceTypeBytes() {
    java.lang.Object ref = confidenceType_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      confidenceType_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  private byte memoizedIsInitialized = -1;
  @java.lang.Override
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  @java.lang.Override
  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) != 0)) {
      output.writeMessage(1, getMetadata());
    }
    if (((bitField0_ & 0x00000002) != 0)) {
      output.writeMessage(2, getParams());
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(confidenceType_)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 3, confidenceType_);
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) != 0)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getMetadata());
    }
    if (((bitField0_ & 0x00000002) != 0)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, getParams());
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(confidenceType_)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(3, confidenceType_);
    }
    size += getUnknownFields().getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.tribuo.classification.sgd.protos.CRFModelProto)) {
      return super.equals(obj);
    }
    org.tribuo.classification.sgd.protos.CRFModelProto other = (org.tribuo.classification.sgd.protos.CRFModelProto) obj;

    if (hasMetadata() != other.hasMetadata()) return false;
    if (hasMetadata()) {
      if (!getMetadata()
          .equals(other.getMetadata())) return false;
    }
    if (hasParams() != other.hasParams()) return false;
    if (hasParams()) {
      if (!getParams()
          .equals(other.getParams())) return false;
    }
    if (!getConfidenceType()
        .equals(other.getConfidenceType())) return false;
    if (!getUnknownFields().equals(other.getUnknownFields())) return false;
    return true;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    if (hasMetadata()) {
      hash = (37 * hash) + METADATA_FIELD_NUMBER;
      hash = (53 * hash) + getMetadata().hashCode();
    }
    if (hasParams()) {
      hash = (37 * hash) + PARAMS_FIELD_NUMBER;
      hash = (53 * hash) + getParams().hashCode();
    }
    hash = (37 * hash) + CONFIDENCE_TYPE_FIELD_NUMBER;
    hash = (53 * hash) + getConfidenceType().hashCode();
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.classification.sgd.protos.CRFModelProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.classification.sgd.protos.CRFModelProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.classification.sgd.protos.CRFModelProto parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  @java.lang.Override
  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.tribuo.classification.sgd.protos.CRFModelProto prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  @java.lang.Override
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * <pre>
   *
   *CRFModel proto
   * </pre>
   *
   * Protobuf type {@code tribuo.classification.sgd.CRFModelProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.classification.sgd.CRFModelProto)
      org.tribuo.classification.sgd.protos.CRFModelProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_CRFModelProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_CRFModelProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.classification.sgd.protos.CRFModelProto.class, org.tribuo.classification.sgd.protos.CRFModelProto.Builder.class);
    }

    // Construct using org.tribuo.classification.sgd.protos.CRFModelProto.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
        getMetadataFieldBuilder();
        getParamsFieldBuilder();
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      bitField0_ = 0;
      metadata_ = null;
      if (metadataBuilder_ != null) {
        metadataBuilder_.dispose();
        metadataBuilder_ = null;
      }
      params_ = null;
      if (paramsBuilder_ != null) {
        paramsBuilder_.dispose();
        paramsBuilder_ = null;
      }
      confidenceType_ = "";
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_CRFModelProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.classification.sgd.protos.CRFModelProto getDefaultInstanceForType() {
      return org.tribuo.classification.sgd.protos.CRFModelProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.classification.sgd.protos.CRFModelProto build() {
      org.tribuo.classification.sgd.protos.CRFModelProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.classification.sgd.protos.CRFModelProto buildPartial() {
      org.tribuo.classification.sgd.protos.CRFModelProto result = new org.tribuo.classification.sgd.protos.CRFModelProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.classification.sgd.protos.CRFModelProto result) {
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.metadata_ = metadataBuilder_ == null
            ? metadata_
            : metadataBuilder_.build();
        to_bitField0_ |= 0x00000001;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        result.params_ = paramsBuilder_ == null
            ? params_
            : paramsBuilder_.build();
        to_bitField0_ |= 0x00000002;
      }
      if (((from_bitField0_ & 0x00000004) != 0)) {
        result.confidenceType_ = confidenceType_;
      }
      result.bitField0_ |= to_bitField0_;
    }

    @java.lang.Override
    public Builder clone() {
      return super.clone();
    }
    @java.lang.Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.setField(field, value);
    }
    @java.lang.Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return super.clearField(field);
    }
    @java.lang.Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return super.clearOneof(oneof);
    }
    @java.lang.Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return super.setRepeatedField(field, index, value);
    }
    @java.lang.Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.addRepeatedField(field, value);
    }
    @java.lang.Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.tribuo.classification.sgd.protos.CRFModelProto) {
        return mergeFrom((org.tribuo.classification.sgd.protos.CRFModelProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.classification.sgd.protos.CRFModelProto other) {
      if (other == org.tribuo.classification.sgd.protos.CRFModelProto.getDefaultInstance()) return this;
      if (other.hasMetadata()) {
        mergeMetadata(other.getMetadata());
      }
      if (other.hasParams()) {
        mergeParams(other.getParams());
      }
      if (!other.getConfidenceType().isEmpty()) {
        confidenceType_ = other.confidenceType_;
        bitField0_ |= 0x00000004;
        onChanged();
      }
      this.mergeUnknownFields(other.getUnknownFields());
      onChanged();
      return this;
    }

    @java.lang.Override
    public final boolean isInitialized() {
      return true;
    }

    @java.lang.Override
    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      if (extensionRegistry == null) {
        throw new java.lang.NullPointerException();
      }
      try {
        boolean done = false;
        while (!done) {
          int tag = input.readTag();
          switch (tag) {
            case 0:
              done = true;
              break;
            case 10: {
              input.readMessage(
                  getMetadataFieldBuilder().getBuilder(),
                  extensionRegistry);
              bitField0_ |= 0x00000001;
              break;
            } // case 10
            case 18: {
              input.readMessage(
                  getParamsFieldBuilder().getBuilder(),
                  extensionRegistry);
              bitField0_ |= 0x00000002;
              break;
            } // case 18
            case 26: {
              confidenceType_ = input.readStringRequireUtf8();
              bitField0_ |= 0x00000004;
              break;
            } // case 26
            default: {
              if (!super.parseUnknownField(input, extensionRegistry, tag)) {
                done = true; // was an endgroup tag
              }
              break;
            } // default:
          } // switch (tag)
        } // while (!done)
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw e.unwrapIOException();
      } finally {
        onChanged();
      } // finally
      return this;
    }
    private int bitField0_;

    private org.tribuo.protos.core.ModelDataProto metadata_;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.protos.core.ModelDataProto, org.tribuo.protos.core.ModelDataProto.Builder, org.tribuo.protos.core.ModelDataProtoOrBuilder> metadataBuilder_;
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     * @return Whether the metadata field is set.
     */
    public boolean hasMetadata() {
      return ((bitField0_ & 0x00000001) != 0);
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     * @return The metadata.
     */
    public org.tribuo.protos.core.ModelDataProto getMetadata() {
      if (metadataBuilder_ == null) {
        return metadata_ == null ? org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
      } else {
        return metadataBuilder_.getMessage();
      }
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder setMetadata(org.tribuo.protos.core.ModelDataProto value) {
      if (metadataBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        metadata_ = value;
      } else {
        metadataBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder setMetadata(
        org.tribuo.protos.core.ModelDataProto.Builder builderForValue) {
      if (metadataBuilder_ == null) {
        metadata_ = builderForValue.build();
      } else {
        metadataBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder mergeMetadata(org.tribuo.protos.core.ModelDataProto value) {
      if (metadataBuilder_ == null) {
        if (((bitField0_ & 0x00000001) != 0) &&
          metadata_ != null &&
          metadata_ != org.tribuo.protos.core.ModelDataProto.getDefaultInstance()) {
          getMetadataBuilder().mergeFrom(value);
        } else {
          metadata_ = value;
        }
      } else {
        metadataBuilder_.mergeFrom(value);
      }
      if (metadata_ != null) {
        bitField0_ |= 0x00000001;
        onChanged();
      }
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder clearMetadata() {
      bitField0_ = (bitField0_ & ~0x00000001);
      metadata_ = null;
      if (metadataBuilder_ != null) {
        metadataBuilder_.dispose();
        metadataBuilder_ = null;
      }
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public org.tribuo.protos.core.ModelDataProto.Builder getMetadataBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getMetadataFieldBuilder().getBuilder();
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public org.tribuo.protos.core.ModelDataProtoOrBuilder getMetadataOrBuilder() {
      if (metadataBuilder_ != null) {
        return metadataBuilder_.getMessageOrBuilder();
      } else {
        return metadata_ == null ?
            org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
      }
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.protos.core.ModelDataProto, org.tribuo.protos.core.ModelDataProto.Builder, org.tribuo.protos.core.ModelDataProtoOrBuilder> 
        getMetadataFieldBuilder() {
      if (metadataBuilder_ == null) {
        metadataBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.tribuo.protos.core.ModelDataProto, org.tribuo.protos.core.ModelDataProto.Builder, org.tribuo.protos.core.ModelDataProtoOrBuilder>(
                getMetadata(),
                getParentForChildren(),
                isClean());
        metadata_ = null;
      }
      return metadataBuilder_;
    }

    private org.tribuo.math.protos.ParametersProto params_;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.math.protos.ParametersProto, org.tribuo.math.protos.ParametersProto.Builder, org.tribuo.math.protos.ParametersProtoOrBuilder> paramsBuilder_;
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     * @return Whether the params field is set.
     */
    public boolean hasParams() {
      return ((bitField0_ & 0x00000002) != 0);
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     * @return The params.
     */
    public org.tribuo.math.protos.ParametersProto getParams() {
      if (paramsBuilder_ == null) {
        return params_ == null ? org.tribuo.math.protos.ParametersProto.getDefaultInstance() : params_;
      } else {
        return paramsBuilder_.getMessage();
      }
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     */
    public Builder setParams(org.tribuo.math.protos.ParametersProto value) {
      if (paramsBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        params_ = value;
      } else {
        paramsBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     */
    public Builder setParams(
        org.tribuo.math.protos.ParametersProto.Builder builderForValue) {
      if (paramsBuilder_ == null) {
        params_ = builderForValue.build();
      } else {
        paramsBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     */
    public Builder mergeParams(org.tribuo.math.protos.ParametersProto value) {
      if (paramsBuilder_ == null) {
        if (((bitField0_ & 0x00000002) != 0) &&
          params_ != null &&
          params_ != org.tribuo.math.protos.ParametersProto.getDefaultInstance()) {
          getParamsBuilder().mergeFrom(value);
        } else {
          params_ = value;
        }
      } else {
        paramsBuilder_.mergeFrom(value);
      }
      if (params_ != null) {
        bitField0_ |= 0x00000002;
        onChanged();
      }
      return this;
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     */
    public Builder clearParams() {
      bitField0_ = (bitField0_ & ~0x00000002);
      params_ = null;
      if (paramsBuilder_ != null) {
        paramsBuilder_.dispose();
        paramsBuilder_ = null;
      }
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     */
    public org.tribuo.math.protos.ParametersProto.Builder getParamsBuilder() {
      bitField0_ |= 0x00000002;
      onChanged();
      return getParamsFieldBuilder().getBuilder();
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     */
    public org.tribuo.math.protos.ParametersProtoOrBuilder getParamsOrBuilder() {
      if (paramsBuilder_ != null) {
        return paramsBuilder_.getMessageOrBuilder();
      } else {
        return params_ == null ?
            org.tribuo.math.protos.ParametersProto.getDefaultInstance() : params_;
      }
    }
    /**
     * <code>.tribuo.math.ParametersProto params = 2;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.math.protos.ParametersProto, org.tribuo.math.protos.ParametersProto.Builder, org.tribuo.math.protos.ParametersProtoOrBuilder> 
        getParamsFieldBuilder() {
      if (paramsBuilder_ == null) {
        paramsBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.tribuo.math.protos.ParametersProto, org.tribuo.math.protos.ParametersProto.Builder, org.tribuo.math.protos.ParametersProtoOrBuilder>(
                getParams(),
                getParentForChildren(),
                isClean());
        params_ = null;
      }
      return paramsBuilder_;
    }

    private java.lang.Object confidenceType_ = "";
    /**
     * <code>string confidence_type = 3;</code>
     * @return The confidenceType.
     */
    public java.lang.String getConfidenceType() {
      java.lang.Object ref = confidenceType_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        confidenceType_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string confidence_type = 3;</code>
     * @return The bytes for confidenceType.
     */
    public com.google.protobuf.ByteString
        getConfidenceTypeBytes() {
      java.lang.Object ref = confidenceType_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        confidenceType_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string confidence_type = 3;</code>
     * @param value The confidenceType to set.
     * @return This builder for chaining.
     */
    public Builder setConfidenceType(
        java.lang.String value) {
      if (value == null) { throw new NullPointerException(); }
      confidenceType_ = value;
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>string confidence_type = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearConfidenceType() {
      confidenceType_ = getDefaultInstance().getConfidenceType();
      bitField0_ = (bitField0_ & ~0x00000004);
      onChanged();
      return this;
    }
    /**
     * <code>string confidence_type = 3;</code>
     * @param value The bytes for confidenceType to set.
     * @return This builder for chaining.
     */
    public Builder setConfidenceTypeBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) { throw new NullPointerException(); }
      checkByteStringIsUtf8(value);
      confidenceType_ = value;
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    @java.lang.Override
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    @java.lang.Override
    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:tribuo.classification.sgd.CRFModelProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.classification.sgd.CRFModelProto)
  private static final org.tribuo.classification.sgd.protos.CRFModelProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.classification.sgd.protos.CRFModelProto();
  }

  public static org.tribuo.classification.sgd.protos.CRFModelProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<CRFModelProto>
      PARSER = new com.google.protobuf.AbstractParser<CRFModelProto>() {
    @java.lang.Override
    public CRFModelProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      Builder builder = newBuilder();
      try {
        builder.mergeFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw e.setUnfinishedMessage(builder.buildPartial());
      } catch (com.google.protobuf.UninitializedMessageException e) {
        throw e.asInvalidProtocolBufferException().setUnfinishedMessage(builder.buildPartial());
      } catch (java.io.IOException e) {
        throw new com.google.protobuf.InvalidProtocolBufferException(e)
            .setUnfinishedMessage(builder.buildPartial());
      }
      return builder.buildPartial();
    }
  };

  public static com.google.protobuf.Parser<CRFModelProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<CRFModelProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.classification.sgd.protos.CRFModelProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


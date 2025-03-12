// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-classification-sgd.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.classification.sgd.protos;

/**
 * <pre>
 *
 *LinearSGDModel proto
 * </pre>
 *
 * Protobuf type {@code tribuo.classification.sgd.ClassificationLinearSGDProto}
 */
public final class ClassificationLinearSGDProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.classification.sgd.ClassificationLinearSGDProto)
    ClassificationLinearSGDProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use ClassificationLinearSGDProto.newBuilder() to construct.
  private ClassificationLinearSGDProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ClassificationLinearSGDProto() {
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new ClassificationLinearSGDProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto.class, org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto.Builder.class);
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

  public static final int NORMALIZER_FIELD_NUMBER = 3;
  private org.tribuo.math.protos.NormalizerProto normalizer_;
  /**
   * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
   * @return Whether the normalizer field is set.
   */
  @java.lang.Override
  public boolean hasNormalizer() {
    return ((bitField0_ & 0x00000004) != 0);
  }
  /**
   * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
   * @return The normalizer.
   */
  @java.lang.Override
  public org.tribuo.math.protos.NormalizerProto getNormalizer() {
    return normalizer_ == null ? org.tribuo.math.protos.NormalizerProto.getDefaultInstance() : normalizer_;
  }
  /**
   * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
   */
  @java.lang.Override
  public org.tribuo.math.protos.NormalizerProtoOrBuilder getNormalizerOrBuilder() {
    return normalizer_ == null ? org.tribuo.math.protos.NormalizerProto.getDefaultInstance() : normalizer_;
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
    if (((bitField0_ & 0x00000004) != 0)) {
      output.writeMessage(3, getNormalizer());
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
    if (((bitField0_ & 0x00000004) != 0)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(3, getNormalizer());
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
    if (!(obj instanceof org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto)) {
      return super.equals(obj);
    }
    org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto other = (org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto) obj;

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
    if (hasNormalizer() != other.hasNormalizer()) return false;
    if (hasNormalizer()) {
      if (!getNormalizer()
          .equals(other.getNormalizer())) return false;
    }
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
    if (hasNormalizer()) {
      hash = (37 * hash) + NORMALIZER_FIELD_NUMBER;
      hash = (53 * hash) + getNormalizer().hashCode();
    }
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto prototype) {
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
   *LinearSGDModel proto
   * </pre>
   *
   * Protobuf type {@code tribuo.classification.sgd.ClassificationLinearSGDProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.classification.sgd.ClassificationLinearSGDProto)
      org.tribuo.classification.sgd.protos.ClassificationLinearSGDProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto.class, org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto.Builder.class);
    }

    // Construct using org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto.newBuilder()
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
        getNormalizerFieldBuilder();
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
      normalizer_ = null;
      if (normalizerBuilder_ != null) {
        normalizerBuilder_.dispose();
        normalizerBuilder_ = null;
      }
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.classification.sgd.protos.TribuoClassificationSgd.internal_static_tribuo_classification_sgd_ClassificationLinearSGDProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto getDefaultInstanceForType() {
      return org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto build() {
      org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto buildPartial() {
      org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto result = new org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto result) {
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
        result.normalizer_ = normalizerBuilder_ == null
            ? normalizer_
            : normalizerBuilder_.build();
        to_bitField0_ |= 0x00000004;
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
      if (other instanceof org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto) {
        return mergeFrom((org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto other) {
      if (other == org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto.getDefaultInstance()) return this;
      if (other.hasMetadata()) {
        mergeMetadata(other.getMetadata());
      }
      if (other.hasParams()) {
        mergeParams(other.getParams());
      }
      if (other.hasNormalizer()) {
        mergeNormalizer(other.getNormalizer());
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
              input.readMessage(
                  getNormalizerFieldBuilder().getBuilder(),
                  extensionRegistry);
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

    private org.tribuo.math.protos.NormalizerProto normalizer_;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.math.protos.NormalizerProto, org.tribuo.math.protos.NormalizerProto.Builder, org.tribuo.math.protos.NormalizerProtoOrBuilder> normalizerBuilder_;
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     * @return Whether the normalizer field is set.
     */
    public boolean hasNormalizer() {
      return ((bitField0_ & 0x00000004) != 0);
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     * @return The normalizer.
     */
    public org.tribuo.math.protos.NormalizerProto getNormalizer() {
      if (normalizerBuilder_ == null) {
        return normalizer_ == null ? org.tribuo.math.protos.NormalizerProto.getDefaultInstance() : normalizer_;
      } else {
        return normalizerBuilder_.getMessage();
      }
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     */
    public Builder setNormalizer(org.tribuo.math.protos.NormalizerProto value) {
      if (normalizerBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        normalizer_ = value;
      } else {
        normalizerBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     */
    public Builder setNormalizer(
        org.tribuo.math.protos.NormalizerProto.Builder builderForValue) {
      if (normalizerBuilder_ == null) {
        normalizer_ = builderForValue.build();
      } else {
        normalizerBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     */
    public Builder mergeNormalizer(org.tribuo.math.protos.NormalizerProto value) {
      if (normalizerBuilder_ == null) {
        if (((bitField0_ & 0x00000004) != 0) &&
          normalizer_ != null &&
          normalizer_ != org.tribuo.math.protos.NormalizerProto.getDefaultInstance()) {
          getNormalizerBuilder().mergeFrom(value);
        } else {
          normalizer_ = value;
        }
      } else {
        normalizerBuilder_.mergeFrom(value);
      }
      if (normalizer_ != null) {
        bitField0_ |= 0x00000004;
        onChanged();
      }
      return this;
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     */
    public Builder clearNormalizer() {
      bitField0_ = (bitField0_ & ~0x00000004);
      normalizer_ = null;
      if (normalizerBuilder_ != null) {
        normalizerBuilder_.dispose();
        normalizerBuilder_ = null;
      }
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     */
    public org.tribuo.math.protos.NormalizerProto.Builder getNormalizerBuilder() {
      bitField0_ |= 0x00000004;
      onChanged();
      return getNormalizerFieldBuilder().getBuilder();
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     */
    public org.tribuo.math.protos.NormalizerProtoOrBuilder getNormalizerOrBuilder() {
      if (normalizerBuilder_ != null) {
        return normalizerBuilder_.getMessageOrBuilder();
      } else {
        return normalizer_ == null ?
            org.tribuo.math.protos.NormalizerProto.getDefaultInstance() : normalizer_;
      }
    }
    /**
     * <code>.tribuo.math.NormalizerProto normalizer = 3;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.math.protos.NormalizerProto, org.tribuo.math.protos.NormalizerProto.Builder, org.tribuo.math.protos.NormalizerProtoOrBuilder> 
        getNormalizerFieldBuilder() {
      if (normalizerBuilder_ == null) {
        normalizerBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.tribuo.math.protos.NormalizerProto, org.tribuo.math.protos.NormalizerProto.Builder, org.tribuo.math.protos.NormalizerProtoOrBuilder>(
                getNormalizer(),
                getParentForChildren(),
                isClean());
        normalizer_ = null;
      }
      return normalizerBuilder_;
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


    // @@protoc_insertion_point(builder_scope:tribuo.classification.sgd.ClassificationLinearSGDProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.classification.sgd.ClassificationLinearSGDProto)
  private static final org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto();
  }

  public static org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<ClassificationLinearSGDProto>
      PARSER = new com.google.protobuf.AbstractParser<ClassificationLinearSGDProto>() {
    @java.lang.Override
    public ClassificationLinearSGDProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<ClassificationLinearSGDProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ClassificationLinearSGDProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.classification.sgd.protos.ClassificationLinearSGDProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-onnx.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.interop.onnx.protos;

/**
 * <pre>
 *
 *LabelTransformer proto
 * </pre>
 *
 * Protobuf type {@code tribuo.interop.onnx.LabelTransformerProto}
 */
public final class LabelTransformerProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.interop.onnx.LabelTransformerProto)
    LabelTransformerProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use LabelTransformerProto.newBuilder() to construct.
  private LabelTransformerProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LabelTransformerProto() {
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new LabelTransformerProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.interop.onnx.protos.TribuoOnnx.internal_static_tribuo_interop_onnx_LabelTransformerProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.interop.onnx.protos.TribuoOnnx.internal_static_tribuo_interop_onnx_LabelTransformerProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.interop.onnx.protos.LabelTransformerProto.class, org.tribuo.interop.onnx.protos.LabelTransformerProto.Builder.class);
  }

  public static final int GENERATES_PROBABILITIES_FIELD_NUMBER = 1;
  private boolean generatesProbabilities_ = false;
  /**
   * <code>bool generates_probabilities = 1;</code>
   * @return The generatesProbabilities.
   */
  @java.lang.Override
  public boolean getGeneratesProbabilities() {
    return generatesProbabilities_;
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
    if (generatesProbabilities_ != false) {
      output.writeBool(1, generatesProbabilities_);
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (generatesProbabilities_ != false) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(1, generatesProbabilities_);
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
    if (!(obj instanceof org.tribuo.interop.onnx.protos.LabelTransformerProto)) {
      return super.equals(obj);
    }
    org.tribuo.interop.onnx.protos.LabelTransformerProto other = (org.tribuo.interop.onnx.protos.LabelTransformerProto) obj;

    if (getGeneratesProbabilities()
        != other.getGeneratesProbabilities()) return false;
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
    hash = (37 * hash) + GENERATES_PROBABILITIES_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
        getGeneratesProbabilities());
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.interop.onnx.protos.LabelTransformerProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.interop.onnx.protos.LabelTransformerProto prototype) {
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
   *LabelTransformer proto
   * </pre>
   *
   * Protobuf type {@code tribuo.interop.onnx.LabelTransformerProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.interop.onnx.LabelTransformerProto)
      org.tribuo.interop.onnx.protos.LabelTransformerProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.interop.onnx.protos.TribuoOnnx.internal_static_tribuo_interop_onnx_LabelTransformerProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.interop.onnx.protos.TribuoOnnx.internal_static_tribuo_interop_onnx_LabelTransformerProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.interop.onnx.protos.LabelTransformerProto.class, org.tribuo.interop.onnx.protos.LabelTransformerProto.Builder.class);
    }

    // Construct using org.tribuo.interop.onnx.protos.LabelTransformerProto.newBuilder()
    private Builder() {

    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);

    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      bitField0_ = 0;
      generatesProbabilities_ = false;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.interop.onnx.protos.TribuoOnnx.internal_static_tribuo_interop_onnx_LabelTransformerProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.interop.onnx.protos.LabelTransformerProto getDefaultInstanceForType() {
      return org.tribuo.interop.onnx.protos.LabelTransformerProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.interop.onnx.protos.LabelTransformerProto build() {
      org.tribuo.interop.onnx.protos.LabelTransformerProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.interop.onnx.protos.LabelTransformerProto buildPartial() {
      org.tribuo.interop.onnx.protos.LabelTransformerProto result = new org.tribuo.interop.onnx.protos.LabelTransformerProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.interop.onnx.protos.LabelTransformerProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.generatesProbabilities_ = generatesProbabilities_;
      }
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
      if (other instanceof org.tribuo.interop.onnx.protos.LabelTransformerProto) {
        return mergeFrom((org.tribuo.interop.onnx.protos.LabelTransformerProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.interop.onnx.protos.LabelTransformerProto other) {
      if (other == org.tribuo.interop.onnx.protos.LabelTransformerProto.getDefaultInstance()) return this;
      if (other.getGeneratesProbabilities() != false) {
        setGeneratesProbabilities(other.getGeneratesProbabilities());
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
            case 8: {
              generatesProbabilities_ = input.readBool();
              bitField0_ |= 0x00000001;
              break;
            } // case 8
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

    private boolean generatesProbabilities_ ;
    /**
     * <code>bool generates_probabilities = 1;</code>
     * @return The generatesProbabilities.
     */
    @java.lang.Override
    public boolean getGeneratesProbabilities() {
      return generatesProbabilities_;
    }
    /**
     * <code>bool generates_probabilities = 1;</code>
     * @param value The generatesProbabilities to set.
     * @return This builder for chaining.
     */
    public Builder setGeneratesProbabilities(boolean value) {

      generatesProbabilities_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>bool generates_probabilities = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearGeneratesProbabilities() {
      bitField0_ = (bitField0_ & ~0x00000001);
      generatesProbabilities_ = false;
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


    // @@protoc_insertion_point(builder_scope:tribuo.interop.onnx.LabelTransformerProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.interop.onnx.LabelTransformerProto)
  private static final org.tribuo.interop.onnx.protos.LabelTransformerProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.interop.onnx.protos.LabelTransformerProto();
  }

  public static org.tribuo.interop.onnx.protos.LabelTransformerProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<LabelTransformerProto>
      PARSER = new com.google.protobuf.AbstractParser<LabelTransformerProto>() {
    @java.lang.Override
    public LabelTransformerProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<LabelTransformerProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LabelTransformerProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.interop.onnx.protos.LabelTransformerProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


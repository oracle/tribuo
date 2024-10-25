// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-oci.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.interop.oci.protos;

/**
 * <pre>
 *
 *OCIMultiLabelConverter proto
 * </pre>
 *
 * Protobuf type {@code tribuo.interop.oci.OCIMultiLabelConverterProto}
 */
public final class OCIMultiLabelConverterProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.interop.oci.OCIMultiLabelConverterProto)
    OCIMultiLabelConverterProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use OCIMultiLabelConverterProto.newBuilder() to construct.
  private OCIMultiLabelConverterProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private OCIMultiLabelConverterProto() {
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new OCIMultiLabelConverterProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.interop.oci.protos.TribuoOci.internal_static_tribuo_interop_oci_OCIMultiLabelConverterProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.interop.oci.protos.TribuoOci.internal_static_tribuo_interop_oci_OCIMultiLabelConverterProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto.class, org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto.Builder.class);
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

  public static final int THRESHOLD_FIELD_NUMBER = 2;
  private double threshold_ = 0D;
  /**
   * <code>double threshold = 2;</code>
   * @return The threshold.
   */
  @java.lang.Override
  public double getThreshold() {
    return threshold_;
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
    if (java.lang.Double.doubleToRawLongBits(threshold_) != 0) {
      output.writeDouble(2, threshold_);
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
    if (java.lang.Double.doubleToRawLongBits(threshold_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(2, threshold_);
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
    if (!(obj instanceof org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto)) {
      return super.equals(obj);
    }
    org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto other = (org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto) obj;

    if (getGeneratesProbabilities()
        != other.getGeneratesProbabilities()) return false;
    if (java.lang.Double.doubleToLongBits(getThreshold())
        != java.lang.Double.doubleToLongBits(
            other.getThreshold())) return false;
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
    hash = (37 * hash) + THRESHOLD_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getThreshold()));
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto prototype) {
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
   *OCIMultiLabelConverter proto
   * </pre>
   *
   * Protobuf type {@code tribuo.interop.oci.OCIMultiLabelConverterProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.interop.oci.OCIMultiLabelConverterProto)
      org.tribuo.interop.oci.protos.OCIMultiLabelConverterProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.interop.oci.protos.TribuoOci.internal_static_tribuo_interop_oci_OCIMultiLabelConverterProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.interop.oci.protos.TribuoOci.internal_static_tribuo_interop_oci_OCIMultiLabelConverterProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto.class, org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto.Builder.class);
    }

    // Construct using org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto.newBuilder()
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
      threshold_ = 0D;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.interop.oci.protos.TribuoOci.internal_static_tribuo_interop_oci_OCIMultiLabelConverterProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto getDefaultInstanceForType() {
      return org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto build() {
      org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto buildPartial() {
      org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto result = new org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.generatesProbabilities_ = generatesProbabilities_;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        result.threshold_ = threshold_;
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
      if (other instanceof org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto) {
        return mergeFrom((org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto other) {
      if (other == org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto.getDefaultInstance()) return this;
      if (other.getGeneratesProbabilities() != false) {
        setGeneratesProbabilities(other.getGeneratesProbabilities());
      }
      if (other.getThreshold() != 0D) {
        setThreshold(other.getThreshold());
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
            case 17: {
              threshold_ = input.readDouble();
              bitField0_ |= 0x00000002;
              break;
            } // case 17
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

    private double threshold_ ;
    /**
     * <code>double threshold = 2;</code>
     * @return The threshold.
     */
    @java.lang.Override
    public double getThreshold() {
      return threshold_;
    }
    /**
     * <code>double threshold = 2;</code>
     * @param value The threshold to set.
     * @return This builder for chaining.
     */
    public Builder setThreshold(double value) {

      threshold_ = value;
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>double threshold = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearThreshold() {
      bitField0_ = (bitField0_ & ~0x00000002);
      threshold_ = 0D;
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


    // @@protoc_insertion_point(builder_scope:tribuo.interop.oci.OCIMultiLabelConverterProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.interop.oci.OCIMultiLabelConverterProto)
  private static final org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto();
  }

  public static org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<OCIMultiLabelConverterProto>
      PARSER = new com.google.protobuf.AbstractParser<OCIMultiLabelConverterProto>() {
    @java.lang.Override
    public OCIMultiLabelConverterProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<OCIMultiLabelConverterProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<OCIMultiLabelConverterProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.interop.oci.protos.OCIMultiLabelConverterProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


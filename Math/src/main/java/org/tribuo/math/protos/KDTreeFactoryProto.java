// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-math-impl.proto

package org.tribuo.math.protos;

/**
 * <pre>
 *KDTreeFactory proto
 * </pre>
 *
 * Protobuf type {@code tribuo.core.KDTreeFactoryProto}
 */
public final class KDTreeFactoryProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.core.KDTreeFactoryProto)
    KDTreeFactoryProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use KDTreeFactoryProto.newBuilder() to construct.
  private KDTreeFactoryProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private KDTreeFactoryProto() {
    distanceType_ = "";
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new KDTreeFactoryProto();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private KDTreeFactoryProto(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          case 8: {

            numThreads_ = input.readInt32();
            break;
          }
          case 18: {
            java.lang.String s = input.readStringRequireUtf8();

            distanceType_ = s;
            break;
          }
          default: {
            if (!parseUnknownField(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.math.protos.TribuoMathImpl.internal_static_tribuo_core_KDTreeFactoryProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.math.protos.TribuoMathImpl.internal_static_tribuo_core_KDTreeFactoryProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.math.protos.KDTreeFactoryProto.class, org.tribuo.math.protos.KDTreeFactoryProto.Builder.class);
  }

  public static final int NUMTHREADS_FIELD_NUMBER = 1;
  private int numThreads_;
  /**
   * <code>int32 numThreads = 1;</code>
   * @return The numThreads.
   */
  @java.lang.Override
  public int getNumThreads() {
    return numThreads_;
  }

  public static final int DISTANCETYPE_FIELD_NUMBER = 2;
  private volatile java.lang.Object distanceType_;
  /**
   * <code>string distanceType = 2;</code>
   * @return The distanceType.
   */
  @java.lang.Override
  public java.lang.String getDistanceType() {
    java.lang.Object ref = distanceType_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      distanceType_ = s;
      return s;
    }
  }
  /**
   * <code>string distanceType = 2;</code>
   * @return The bytes for distanceType.
   */
  @java.lang.Override
  public com.google.protobuf.ByteString
      getDistanceTypeBytes() {
    java.lang.Object ref = distanceType_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      distanceType_ = b;
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
    if (numThreads_ != 0) {
      output.writeInt32(1, numThreads_);
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(distanceType_)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, distanceType_);
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (numThreads_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(1, numThreads_);
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(distanceType_)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, distanceType_);
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.tribuo.math.protos.KDTreeFactoryProto)) {
      return super.equals(obj);
    }
    org.tribuo.math.protos.KDTreeFactoryProto other = (org.tribuo.math.protos.KDTreeFactoryProto) obj;

    if (getNumThreads()
        != other.getNumThreads()) return false;
    if (!getDistanceType()
        .equals(other.getDistanceType())) return false;
    if (!unknownFields.equals(other.unknownFields)) return false;
    return true;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    hash = (37 * hash) + NUMTHREADS_FIELD_NUMBER;
    hash = (53 * hash) + getNumThreads();
    hash = (37 * hash) + DISTANCETYPE_FIELD_NUMBER;
    hash = (53 * hash) + getDistanceType().hashCode();
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.math.protos.KDTreeFactoryProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.math.protos.KDTreeFactoryProto prototype) {
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
   *KDTreeFactory proto
   * </pre>
   *
   * Protobuf type {@code tribuo.core.KDTreeFactoryProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.core.KDTreeFactoryProto)
      org.tribuo.math.protos.KDTreeFactoryProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.math.protos.TribuoMathImpl.internal_static_tribuo_core_KDTreeFactoryProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.math.protos.TribuoMathImpl.internal_static_tribuo_core_KDTreeFactoryProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.math.protos.KDTreeFactoryProto.class, org.tribuo.math.protos.KDTreeFactoryProto.Builder.class);
    }

    // Construct using org.tribuo.math.protos.KDTreeFactoryProto.newBuilder()
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
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      numThreads_ = 0;

      distanceType_ = "";

      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.math.protos.TribuoMathImpl.internal_static_tribuo_core_KDTreeFactoryProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.math.protos.KDTreeFactoryProto getDefaultInstanceForType() {
      return org.tribuo.math.protos.KDTreeFactoryProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.math.protos.KDTreeFactoryProto build() {
      org.tribuo.math.protos.KDTreeFactoryProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.math.protos.KDTreeFactoryProto buildPartial() {
      org.tribuo.math.protos.KDTreeFactoryProto result = new org.tribuo.math.protos.KDTreeFactoryProto(this);
      result.numThreads_ = numThreads_;
      result.distanceType_ = distanceType_;
      onBuilt();
      return result;
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
      if (other instanceof org.tribuo.math.protos.KDTreeFactoryProto) {
        return mergeFrom((org.tribuo.math.protos.KDTreeFactoryProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.math.protos.KDTreeFactoryProto other) {
      if (other == org.tribuo.math.protos.KDTreeFactoryProto.getDefaultInstance()) return this;
      if (other.getNumThreads() != 0) {
        setNumThreads(other.getNumThreads());
      }
      if (!other.getDistanceType().isEmpty()) {
        distanceType_ = other.distanceType_;
        onChanged();
      }
      this.mergeUnknownFields(other.unknownFields);
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
      org.tribuo.math.protos.KDTreeFactoryProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tribuo.math.protos.KDTreeFactoryProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }

    private int numThreads_ ;
    /**
     * <code>int32 numThreads = 1;</code>
     * @return The numThreads.
     */
    @java.lang.Override
    public int getNumThreads() {
      return numThreads_;
    }
    /**
     * <code>int32 numThreads = 1;</code>
     * @param value The numThreads to set.
     * @return This builder for chaining.
     */
    public Builder setNumThreads(int value) {
      
      numThreads_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 numThreads = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearNumThreads() {
      
      numThreads_ = 0;
      onChanged();
      return this;
    }

    private java.lang.Object distanceType_ = "";
    /**
     * <code>string distanceType = 2;</code>
     * @return The distanceType.
     */
    public java.lang.String getDistanceType() {
      java.lang.Object ref = distanceType_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        distanceType_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string distanceType = 2;</code>
     * @return The bytes for distanceType.
     */
    public com.google.protobuf.ByteString
        getDistanceTypeBytes() {
      java.lang.Object ref = distanceType_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        distanceType_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string distanceType = 2;</code>
     * @param value The distanceType to set.
     * @return This builder for chaining.
     */
    public Builder setDistanceType(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      distanceType_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string distanceType = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearDistanceType() {
      
      distanceType_ = getDefaultInstance().getDistanceType();
      onChanged();
      return this;
    }
    /**
     * <code>string distanceType = 2;</code>
     * @param value The bytes for distanceType to set.
     * @return This builder for chaining.
     */
    public Builder setDistanceTypeBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      distanceType_ = value;
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


    // @@protoc_insertion_point(builder_scope:tribuo.core.KDTreeFactoryProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.core.KDTreeFactoryProto)
  private static final org.tribuo.math.protos.KDTreeFactoryProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.math.protos.KDTreeFactoryProto();
  }

  public static org.tribuo.math.protos.KDTreeFactoryProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<KDTreeFactoryProto>
      PARSER = new com.google.protobuf.AbstractParser<KDTreeFactoryProto>() {
    @java.lang.Override
    public KDTreeFactoryProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new KDTreeFactoryProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<KDTreeFactoryProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<KDTreeFactoryProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.math.protos.KDTreeFactoryProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


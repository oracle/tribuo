// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core.proto

package org.tribuo.protos.core;

/**
 * <pre>
 *Output Factory redirection proto
 * </pre>
 *
 * Protobuf type {@code tribuo.core.OutputFactoryProto}
 */
public final class OutputFactoryProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.core.OutputFactoryProto)
    OutputFactoryProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use OutputFactoryProto.newBuilder() to construct.
  private OutputFactoryProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private OutputFactoryProto() {
    className_ = "";
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new OutputFactoryProto();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private OutputFactoryProto(
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

            version_ = input.readInt32();
            break;
          }
          case 18: {
            java.lang.String s = input.readStringRequireUtf8();

            className_ = s;
            break;
          }
          case 26: {
            com.google.protobuf.Any.Builder subBuilder = null;
            if (serializedData_ != null) {
              subBuilder = serializedData_.toBuilder();
            }
            serializedData_ = input.readMessage(com.google.protobuf.Any.parser(), extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(serializedData_);
              serializedData_ = subBuilder.buildPartial();
            }

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
    return org.tribuo.protos.core.TribuoCore.internal_static_tribuo_core_OutputFactoryProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.protos.core.TribuoCore.internal_static_tribuo_core_OutputFactoryProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.protos.core.OutputFactoryProto.class, org.tribuo.protos.core.OutputFactoryProto.Builder.class);
  }

  public static final int VERSION_FIELD_NUMBER = 1;
  private int version_;
  /**
   * <code>int32 version = 1;</code>
   * @return The version.
   */
  @java.lang.Override
  public int getVersion() {
    return version_;
  }

  public static final int CLASS_NAME_FIELD_NUMBER = 2;
  private volatile java.lang.Object className_;
  /**
   * <code>string class_name = 2;</code>
   * @return The className.
   */
  @java.lang.Override
  public java.lang.String getClassName() {
    java.lang.Object ref = className_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      className_ = s;
      return s;
    }
  }
  /**
   * <code>string class_name = 2;</code>
   * @return The bytes for className.
   */
  @java.lang.Override
  public com.google.protobuf.ByteString
      getClassNameBytes() {
    java.lang.Object ref = className_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      className_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int SERIALIZED_DATA_FIELD_NUMBER = 3;
  private com.google.protobuf.Any serializedData_;
  /**
   * <code>.google.protobuf.Any serialized_data = 3;</code>
   * @return Whether the serializedData field is set.
   */
  @java.lang.Override
  public boolean hasSerializedData() {
    return serializedData_ != null;
  }
  /**
   * <code>.google.protobuf.Any serialized_data = 3;</code>
   * @return The serializedData.
   */
  @java.lang.Override
  public com.google.protobuf.Any getSerializedData() {
    return serializedData_ == null ? com.google.protobuf.Any.getDefaultInstance() : serializedData_;
  }
  /**
   * <code>.google.protobuf.Any serialized_data = 3;</code>
   */
  @java.lang.Override
  public com.google.protobuf.AnyOrBuilder getSerializedDataOrBuilder() {
    return getSerializedData();
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
    if (version_ != 0) {
      output.writeInt32(1, version_);
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(className_)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, className_);
    }
    if (serializedData_ != null) {
      output.writeMessage(3, getSerializedData());
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (version_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(1, version_);
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(className_)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, className_);
    }
    if (serializedData_ != null) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(3, getSerializedData());
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
    if (!(obj instanceof org.tribuo.protos.core.OutputFactoryProto)) {
      return super.equals(obj);
    }
    org.tribuo.protos.core.OutputFactoryProto other = (org.tribuo.protos.core.OutputFactoryProto) obj;

    if (getVersion()
        != other.getVersion()) return false;
    if (!getClassName()
        .equals(other.getClassName())) return false;
    if (hasSerializedData() != other.hasSerializedData()) return false;
    if (hasSerializedData()) {
      if (!getSerializedData()
          .equals(other.getSerializedData())) return false;
    }
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
    hash = (37 * hash) + VERSION_FIELD_NUMBER;
    hash = (53 * hash) + getVersion();
    hash = (37 * hash) + CLASS_NAME_FIELD_NUMBER;
    hash = (53 * hash) + getClassName().hashCode();
    if (hasSerializedData()) {
      hash = (37 * hash) + SERIALIZED_DATA_FIELD_NUMBER;
      hash = (53 * hash) + getSerializedData().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.OutputFactoryProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.protos.core.OutputFactoryProto prototype) {
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
   *Output Factory redirection proto
   * </pre>
   *
   * Protobuf type {@code tribuo.core.OutputFactoryProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.core.OutputFactoryProto)
      org.tribuo.protos.core.OutputFactoryProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.protos.core.TribuoCore.internal_static_tribuo_core_OutputFactoryProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.protos.core.TribuoCore.internal_static_tribuo_core_OutputFactoryProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.protos.core.OutputFactoryProto.class, org.tribuo.protos.core.OutputFactoryProto.Builder.class);
    }

    // Construct using org.tribuo.protos.core.OutputFactoryProto.newBuilder()
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
      version_ = 0;

      className_ = "";

      if (serializedDataBuilder_ == null) {
        serializedData_ = null;
      } else {
        serializedData_ = null;
        serializedDataBuilder_ = null;
      }
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.protos.core.TribuoCore.internal_static_tribuo_core_OutputFactoryProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.protos.core.OutputFactoryProto getDefaultInstanceForType() {
      return org.tribuo.protos.core.OutputFactoryProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.protos.core.OutputFactoryProto build() {
      org.tribuo.protos.core.OutputFactoryProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.protos.core.OutputFactoryProto buildPartial() {
      org.tribuo.protos.core.OutputFactoryProto result = new org.tribuo.protos.core.OutputFactoryProto(this);
      result.version_ = version_;
      result.className_ = className_;
      if (serializedDataBuilder_ == null) {
        result.serializedData_ = serializedData_;
      } else {
        result.serializedData_ = serializedDataBuilder_.build();
      }
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
      if (other instanceof org.tribuo.protos.core.OutputFactoryProto) {
        return mergeFrom((org.tribuo.protos.core.OutputFactoryProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.protos.core.OutputFactoryProto other) {
      if (other == org.tribuo.protos.core.OutputFactoryProto.getDefaultInstance()) return this;
      if (other.getVersion() != 0) {
        setVersion(other.getVersion());
      }
      if (!other.getClassName().isEmpty()) {
        className_ = other.className_;
        onChanged();
      }
      if (other.hasSerializedData()) {
        mergeSerializedData(other.getSerializedData());
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
      org.tribuo.protos.core.OutputFactoryProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tribuo.protos.core.OutputFactoryProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }

    private int version_ ;
    /**
     * <code>int32 version = 1;</code>
     * @return The version.
     */
    @java.lang.Override
    public int getVersion() {
      return version_;
    }
    /**
     * <code>int32 version = 1;</code>
     * @param value The version to set.
     * @return This builder for chaining.
     */
    public Builder setVersion(int value) {
      
      version_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 version = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearVersion() {
      
      version_ = 0;
      onChanged();
      return this;
    }

    private java.lang.Object className_ = "";
    /**
     * <code>string class_name = 2;</code>
     * @return The className.
     */
    public java.lang.String getClassName() {
      java.lang.Object ref = className_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        className_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string class_name = 2;</code>
     * @return The bytes for className.
     */
    public com.google.protobuf.ByteString
        getClassNameBytes() {
      java.lang.Object ref = className_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        className_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string class_name = 2;</code>
     * @param value The className to set.
     * @return This builder for chaining.
     */
    public Builder setClassName(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      className_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string class_name = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearClassName() {
      
      className_ = getDefaultInstance().getClassName();
      onChanged();
      return this;
    }
    /**
     * <code>string class_name = 2;</code>
     * @param value The bytes for className to set.
     * @return This builder for chaining.
     */
    public Builder setClassNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      className_ = value;
      onChanged();
      return this;
    }

    private com.google.protobuf.Any serializedData_;
    private com.google.protobuf.SingleFieldBuilderV3<
        com.google.protobuf.Any, com.google.protobuf.Any.Builder, com.google.protobuf.AnyOrBuilder> serializedDataBuilder_;
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     * @return Whether the serializedData field is set.
     */
    public boolean hasSerializedData() {
      return serializedDataBuilder_ != null || serializedData_ != null;
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     * @return The serializedData.
     */
    public com.google.protobuf.Any getSerializedData() {
      if (serializedDataBuilder_ == null) {
        return serializedData_ == null ? com.google.protobuf.Any.getDefaultInstance() : serializedData_;
      } else {
        return serializedDataBuilder_.getMessage();
      }
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     */
    public Builder setSerializedData(com.google.protobuf.Any value) {
      if (serializedDataBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        serializedData_ = value;
        onChanged();
      } else {
        serializedDataBuilder_.setMessage(value);
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     */
    public Builder setSerializedData(
        com.google.protobuf.Any.Builder builderForValue) {
      if (serializedDataBuilder_ == null) {
        serializedData_ = builderForValue.build();
        onChanged();
      } else {
        serializedDataBuilder_.setMessage(builderForValue.build());
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     */
    public Builder mergeSerializedData(com.google.protobuf.Any value) {
      if (serializedDataBuilder_ == null) {
        if (serializedData_ != null) {
          serializedData_ =
            com.google.protobuf.Any.newBuilder(serializedData_).mergeFrom(value).buildPartial();
        } else {
          serializedData_ = value;
        }
        onChanged();
      } else {
        serializedDataBuilder_.mergeFrom(value);
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     */
    public Builder clearSerializedData() {
      if (serializedDataBuilder_ == null) {
        serializedData_ = null;
        onChanged();
      } else {
        serializedData_ = null;
        serializedDataBuilder_ = null;
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     */
    public com.google.protobuf.Any.Builder getSerializedDataBuilder() {
      
      onChanged();
      return getSerializedDataFieldBuilder().getBuilder();
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     */
    public com.google.protobuf.AnyOrBuilder getSerializedDataOrBuilder() {
      if (serializedDataBuilder_ != null) {
        return serializedDataBuilder_.getMessageOrBuilder();
      } else {
        return serializedData_ == null ?
            com.google.protobuf.Any.getDefaultInstance() : serializedData_;
      }
    }
    /**
     * <code>.google.protobuf.Any serialized_data = 3;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        com.google.protobuf.Any, com.google.protobuf.Any.Builder, com.google.protobuf.AnyOrBuilder> 
        getSerializedDataFieldBuilder() {
      if (serializedDataBuilder_ == null) {
        serializedDataBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            com.google.protobuf.Any, com.google.protobuf.Any.Builder, com.google.protobuf.AnyOrBuilder>(
                getSerializedData(),
                getParentForChildren(),
                isClean());
        serializedData_ = null;
      }
      return serializedDataBuilder_;
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


    // @@protoc_insertion_point(builder_scope:tribuo.core.OutputFactoryProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.core.OutputFactoryProto)
  private static final org.tribuo.protos.core.OutputFactoryProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.protos.core.OutputFactoryProto();
  }

  public static org.tribuo.protos.core.OutputFactoryProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<OutputFactoryProto>
      PARSER = new com.google.protobuf.AbstractParser<OutputFactoryProto>() {
    @java.lang.Override
    public OutputFactoryProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new OutputFactoryProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<OutputFactoryProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<OutputFactoryProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.protos.core.OutputFactoryProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


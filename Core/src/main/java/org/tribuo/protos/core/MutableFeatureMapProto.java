// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

package org.tribuo.protos.core;

/**
 * <pre>
 *MutableFeatureMap proto
 * </pre>
 *
 * Protobuf type {@code tribuo.core.MutableFeatureMapProto}
 */
public final class MutableFeatureMapProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.core.MutableFeatureMapProto)
    MutableFeatureMapProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use MutableFeatureMapProto.newBuilder() to construct.
  private MutableFeatureMapProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private MutableFeatureMapProto() {
    info_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new MutableFeatureMapProto();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private MutableFeatureMapProto(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    int mutable_bitField0_ = 0;
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

            convertHighCardinality_ = input.readBool();
            break;
          }
          case 18: {
            if (!((mutable_bitField0_ & 0x00000001) != 0)) {
              info_ = new java.util.ArrayList<org.tribuo.protos.core.VariableInfoProto>();
              mutable_bitField0_ |= 0x00000001;
            }
            info_.add(
                input.readMessage(org.tribuo.protos.core.VariableInfoProto.parser(), extensionRegistry));
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
      if (((mutable_bitField0_ & 0x00000001) != 0)) {
        info_ = java.util.Collections.unmodifiableList(info_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_MutableFeatureMapProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_MutableFeatureMapProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.protos.core.MutableFeatureMapProto.class, org.tribuo.protos.core.MutableFeatureMapProto.Builder.class);
  }

  public static final int CONVERT_HIGH_CARDINALITY_FIELD_NUMBER = 1;
  private boolean convertHighCardinality_;
  /**
   * <code>bool convert_high_cardinality = 1;</code>
   * @return The convertHighCardinality.
   */
  @java.lang.Override
  public boolean getConvertHighCardinality() {
    return convertHighCardinality_;
  }

  public static final int INFO_FIELD_NUMBER = 2;
  private java.util.List<org.tribuo.protos.core.VariableInfoProto> info_;
  /**
   * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
   */
  @java.lang.Override
  public java.util.List<org.tribuo.protos.core.VariableInfoProto> getInfoList() {
    return info_;
  }
  /**
   * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
   */
  @java.lang.Override
  public java.util.List<? extends org.tribuo.protos.core.VariableInfoProtoOrBuilder> 
      getInfoOrBuilderList() {
    return info_;
  }
  /**
   * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
   */
  @java.lang.Override
  public int getInfoCount() {
    return info_.size();
  }
  /**
   * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
   */
  @java.lang.Override
  public org.tribuo.protos.core.VariableInfoProto getInfo(int index) {
    return info_.get(index);
  }
  /**
   * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
   */
  @java.lang.Override
  public org.tribuo.protos.core.VariableInfoProtoOrBuilder getInfoOrBuilder(
      int index) {
    return info_.get(index);
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
    if (convertHighCardinality_ != false) {
      output.writeBool(1, convertHighCardinality_);
    }
    for (int i = 0; i < info_.size(); i++) {
      output.writeMessage(2, info_.get(i));
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (convertHighCardinality_ != false) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(1, convertHighCardinality_);
    }
    for (int i = 0; i < info_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, info_.get(i));
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
    if (!(obj instanceof org.tribuo.protos.core.MutableFeatureMapProto)) {
      return super.equals(obj);
    }
    org.tribuo.protos.core.MutableFeatureMapProto other = (org.tribuo.protos.core.MutableFeatureMapProto) obj;

    if (getConvertHighCardinality()
        != other.getConvertHighCardinality()) return false;
    if (!getInfoList()
        .equals(other.getInfoList())) return false;
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
    hash = (37 * hash) + CONVERT_HIGH_CARDINALITY_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
        getConvertHighCardinality());
    if (getInfoCount() > 0) {
      hash = (37 * hash) + INFO_FIELD_NUMBER;
      hash = (53 * hash) + getInfoList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.MutableFeatureMapProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.protos.core.MutableFeatureMapProto prototype) {
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
   *MutableFeatureMap proto
   * </pre>
   *
   * Protobuf type {@code tribuo.core.MutableFeatureMapProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.core.MutableFeatureMapProto)
      org.tribuo.protos.core.MutableFeatureMapProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_MutableFeatureMapProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_MutableFeatureMapProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.protos.core.MutableFeatureMapProto.class, org.tribuo.protos.core.MutableFeatureMapProto.Builder.class);
    }

    // Construct using org.tribuo.protos.core.MutableFeatureMapProto.newBuilder()
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
        getInfoFieldBuilder();
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      convertHighCardinality_ = false;

      if (infoBuilder_ == null) {
        info_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        infoBuilder_.clear();
      }
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_MutableFeatureMapProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.protos.core.MutableFeatureMapProto getDefaultInstanceForType() {
      return org.tribuo.protos.core.MutableFeatureMapProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.protos.core.MutableFeatureMapProto build() {
      org.tribuo.protos.core.MutableFeatureMapProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.protos.core.MutableFeatureMapProto buildPartial() {
      org.tribuo.protos.core.MutableFeatureMapProto result = new org.tribuo.protos.core.MutableFeatureMapProto(this);
      int from_bitField0_ = bitField0_;
      result.convertHighCardinality_ = convertHighCardinality_;
      if (infoBuilder_ == null) {
        if (((bitField0_ & 0x00000001) != 0)) {
          info_ = java.util.Collections.unmodifiableList(info_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.info_ = info_;
      } else {
        result.info_ = infoBuilder_.build();
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
      if (other instanceof org.tribuo.protos.core.MutableFeatureMapProto) {
        return mergeFrom((org.tribuo.protos.core.MutableFeatureMapProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.protos.core.MutableFeatureMapProto other) {
      if (other == org.tribuo.protos.core.MutableFeatureMapProto.getDefaultInstance()) return this;
      if (other.getConvertHighCardinality() != false) {
        setConvertHighCardinality(other.getConvertHighCardinality());
      }
      if (infoBuilder_ == null) {
        if (!other.info_.isEmpty()) {
          if (info_.isEmpty()) {
            info_ = other.info_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensureInfoIsMutable();
            info_.addAll(other.info_);
          }
          onChanged();
        }
      } else {
        if (!other.info_.isEmpty()) {
          if (infoBuilder_.isEmpty()) {
            infoBuilder_.dispose();
            infoBuilder_ = null;
            info_ = other.info_;
            bitField0_ = (bitField0_ & ~0x00000001);
            infoBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getInfoFieldBuilder() : null;
          } else {
            infoBuilder_.addAllMessages(other.info_);
          }
        }
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
      org.tribuo.protos.core.MutableFeatureMapProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tribuo.protos.core.MutableFeatureMapProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private boolean convertHighCardinality_ ;
    /**
     * <code>bool convert_high_cardinality = 1;</code>
     * @return The convertHighCardinality.
     */
    @java.lang.Override
    public boolean getConvertHighCardinality() {
      return convertHighCardinality_;
    }
    /**
     * <code>bool convert_high_cardinality = 1;</code>
     * @param value The convertHighCardinality to set.
     * @return This builder for chaining.
     */
    public Builder setConvertHighCardinality(boolean value) {
      
      convertHighCardinality_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>bool convert_high_cardinality = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearConvertHighCardinality() {
      
      convertHighCardinality_ = false;
      onChanged();
      return this;
    }

    private java.util.List<org.tribuo.protos.core.VariableInfoProto> info_ =
      java.util.Collections.emptyList();
    private void ensureInfoIsMutable() {
      if (!((bitField0_ & 0x00000001) != 0)) {
        info_ = new java.util.ArrayList<org.tribuo.protos.core.VariableInfoProto>(info_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.tribuo.protos.core.VariableInfoProto, org.tribuo.protos.core.VariableInfoProto.Builder, org.tribuo.protos.core.VariableInfoProtoOrBuilder> infoBuilder_;

    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public java.util.List<org.tribuo.protos.core.VariableInfoProto> getInfoList() {
      if (infoBuilder_ == null) {
        return java.util.Collections.unmodifiableList(info_);
      } else {
        return infoBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public int getInfoCount() {
      if (infoBuilder_ == null) {
        return info_.size();
      } else {
        return infoBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public org.tribuo.protos.core.VariableInfoProto getInfo(int index) {
      if (infoBuilder_ == null) {
        return info_.get(index);
      } else {
        return infoBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder setInfo(
        int index, org.tribuo.protos.core.VariableInfoProto value) {
      if (infoBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureInfoIsMutable();
        info_.set(index, value);
        onChanged();
      } else {
        infoBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder setInfo(
        int index, org.tribuo.protos.core.VariableInfoProto.Builder builderForValue) {
      if (infoBuilder_ == null) {
        ensureInfoIsMutable();
        info_.set(index, builderForValue.build());
        onChanged();
      } else {
        infoBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder addInfo(org.tribuo.protos.core.VariableInfoProto value) {
      if (infoBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureInfoIsMutable();
        info_.add(value);
        onChanged();
      } else {
        infoBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder addInfo(
        int index, org.tribuo.protos.core.VariableInfoProto value) {
      if (infoBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureInfoIsMutable();
        info_.add(index, value);
        onChanged();
      } else {
        infoBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder addInfo(
        org.tribuo.protos.core.VariableInfoProto.Builder builderForValue) {
      if (infoBuilder_ == null) {
        ensureInfoIsMutable();
        info_.add(builderForValue.build());
        onChanged();
      } else {
        infoBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder addInfo(
        int index, org.tribuo.protos.core.VariableInfoProto.Builder builderForValue) {
      if (infoBuilder_ == null) {
        ensureInfoIsMutable();
        info_.add(index, builderForValue.build());
        onChanged();
      } else {
        infoBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder addAllInfo(
        java.lang.Iterable<? extends org.tribuo.protos.core.VariableInfoProto> values) {
      if (infoBuilder_ == null) {
        ensureInfoIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, info_);
        onChanged();
      } else {
        infoBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder clearInfo() {
      if (infoBuilder_ == null) {
        info_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        infoBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public Builder removeInfo(int index) {
      if (infoBuilder_ == null) {
        ensureInfoIsMutable();
        info_.remove(index);
        onChanged();
      } else {
        infoBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public org.tribuo.protos.core.VariableInfoProto.Builder getInfoBuilder(
        int index) {
      return getInfoFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public org.tribuo.protos.core.VariableInfoProtoOrBuilder getInfoOrBuilder(
        int index) {
      if (infoBuilder_ == null) {
        return info_.get(index);  } else {
        return infoBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public java.util.List<? extends org.tribuo.protos.core.VariableInfoProtoOrBuilder> 
         getInfoOrBuilderList() {
      if (infoBuilder_ != null) {
        return infoBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(info_);
      }
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public org.tribuo.protos.core.VariableInfoProto.Builder addInfoBuilder() {
      return getInfoFieldBuilder().addBuilder(
          org.tribuo.protos.core.VariableInfoProto.getDefaultInstance());
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public org.tribuo.protos.core.VariableInfoProto.Builder addInfoBuilder(
        int index) {
      return getInfoFieldBuilder().addBuilder(
          index, org.tribuo.protos.core.VariableInfoProto.getDefaultInstance());
    }
    /**
     * <code>repeated .tribuo.core.VariableInfoProto info = 2;</code>
     */
    public java.util.List<org.tribuo.protos.core.VariableInfoProto.Builder> 
         getInfoBuilderList() {
      return getInfoFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.tribuo.protos.core.VariableInfoProto, org.tribuo.protos.core.VariableInfoProto.Builder, org.tribuo.protos.core.VariableInfoProtoOrBuilder> 
        getInfoFieldBuilder() {
      if (infoBuilder_ == null) {
        infoBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.tribuo.protos.core.VariableInfoProto, org.tribuo.protos.core.VariableInfoProto.Builder, org.tribuo.protos.core.VariableInfoProtoOrBuilder>(
                info_,
                ((bitField0_ & 0x00000001) != 0),
                getParentForChildren(),
                isClean());
        info_ = null;
      }
      return infoBuilder_;
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


    // @@protoc_insertion_point(builder_scope:tribuo.core.MutableFeatureMapProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.core.MutableFeatureMapProto)
  private static final org.tribuo.protos.core.MutableFeatureMapProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.protos.core.MutableFeatureMapProto();
  }

  public static org.tribuo.protos.core.MutableFeatureMapProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<MutableFeatureMapProto>
      PARSER = new com.google.protobuf.AbstractParser<MutableFeatureMapProto>() {
    @java.lang.Override
    public MutableFeatureMapProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new MutableFeatureMapProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<MutableFeatureMapProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<MutableFeatureMapProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.protos.core.MutableFeatureMapProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


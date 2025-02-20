// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.protos.core;

/**
 * <pre>
 *
 *CategoricalIDInfo proto
 * </pre>
 *
 * Protobuf type {@code tribuo.core.CategoricalIDInfoProto}
 */
public final class CategoricalIDInfoProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.core.CategoricalIDInfoProto)
    CategoricalIDInfoProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use CategoricalIDInfoProto.newBuilder() to construct.
  private CategoricalIDInfoProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private CategoricalIDInfoProto() {
    name_ = "";
    key_ = emptyDoubleList();
    value_ = emptyLongList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new CategoricalIDInfoProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_CategoricalIDInfoProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_CategoricalIDInfoProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.protos.core.CategoricalIDInfoProto.class, org.tribuo.protos.core.CategoricalIDInfoProto.Builder.class);
  }

  public static final int NAME_FIELD_NUMBER = 1;
  @SuppressWarnings("serial")
  private volatile java.lang.Object name_ = "";
  /**
   * <code>string name = 1;</code>
   * @return The name.
   */
  @java.lang.Override
  public java.lang.String getName() {
    java.lang.Object ref = name_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      name_ = s;
      return s;
    }
  }
  /**
   * <code>string name = 1;</code>
   * @return The bytes for name.
   */
  @java.lang.Override
  public com.google.protobuf.ByteString
      getNameBytes() {
    java.lang.Object ref = name_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      name_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int COUNT_FIELD_NUMBER = 2;
  private int count_ = 0;
  /**
   * <code>int32 count = 2;</code>
   * @return The count.
   */
  @java.lang.Override
  public int getCount() {
    return count_;
  }

  public static final int ID_FIELD_NUMBER = 3;
  private int id_ = 0;
  /**
   * <code>int32 id = 3;</code>
   * @return The id.
   */
  @java.lang.Override
  public int getId() {
    return id_;
  }

  public static final int KEY_FIELD_NUMBER = 10;
  @SuppressWarnings("serial")
  private com.google.protobuf.Internal.DoubleList key_ =
      emptyDoubleList();
  /**
   * <code>repeated double key = 10;</code>
   * @return A list containing the key.
   */
  @java.lang.Override
  public java.util.List<java.lang.Double>
      getKeyList() {
    return key_;
  }
  /**
   * <code>repeated double key = 10;</code>
   * @return The count of key.
   */
  public int getKeyCount() {
    return key_.size();
  }
  /**
   * <code>repeated double key = 10;</code>
   * @param index The index of the element to return.
   * @return The key at the given index.
   */
  public double getKey(int index) {
    return key_.getDouble(index);
  }
  private int keyMemoizedSerializedSize = -1;

  public static final int VALUE_FIELD_NUMBER = 11;
  @SuppressWarnings("serial")
  private com.google.protobuf.Internal.LongList value_ =
      emptyLongList();
  /**
   * <code>repeated int64 value = 11;</code>
   * @return A list containing the value.
   */
  @java.lang.Override
  public java.util.List<java.lang.Long>
      getValueList() {
    return value_;
  }
  /**
   * <code>repeated int64 value = 11;</code>
   * @return The count of value.
   */
  public int getValueCount() {
    return value_.size();
  }
  /**
   * <code>repeated int64 value = 11;</code>
   * @param index The index of the element to return.
   * @return The value at the given index.
   */
  public long getValue(int index) {
    return value_.getLong(index);
  }
  private int valueMemoizedSerializedSize = -1;

  public static final int OBSERVED_VALUE_FIELD_NUMBER = 12;
  private double observedValue_ = 0D;
  /**
   * <code>double observed_value = 12;</code>
   * @return The observedValue.
   */
  @java.lang.Override
  public double getObservedValue() {
    return observedValue_;
  }

  public static final int OBSERVED_COUNT_FIELD_NUMBER = 13;
  private long observedCount_ = 0L;
  /**
   * <code>int64 observed_count = 13;</code>
   * @return The observedCount.
   */
  @java.lang.Override
  public long getObservedCount() {
    return observedCount_;
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
    getSerializedSize();
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(name_)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, name_);
    }
    if (count_ != 0) {
      output.writeInt32(2, count_);
    }
    if (id_ != 0) {
      output.writeInt32(3, id_);
    }
    if (getKeyList().size() > 0) {
      output.writeUInt32NoTag(82);
      output.writeUInt32NoTag(keyMemoizedSerializedSize);
    }
    for (int i = 0; i < key_.size(); i++) {
      output.writeDoubleNoTag(key_.getDouble(i));
    }
    if (getValueList().size() > 0) {
      output.writeUInt32NoTag(90);
      output.writeUInt32NoTag(valueMemoizedSerializedSize);
    }
    for (int i = 0; i < value_.size(); i++) {
      output.writeInt64NoTag(value_.getLong(i));
    }
    if (java.lang.Double.doubleToRawLongBits(observedValue_) != 0) {
      output.writeDouble(12, observedValue_);
    }
    if (observedCount_ != 0L) {
      output.writeInt64(13, observedCount_);
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(name_)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, name_);
    }
    if (count_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(2, count_);
    }
    if (id_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(3, id_);
    }
    {
      int dataSize = 0;
      dataSize = 8 * getKeyList().size();
      size += dataSize;
      if (!getKeyList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      keyMemoizedSerializedSize = dataSize;
    }
    {
      int dataSize = 0;
      for (int i = 0; i < value_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeInt64SizeNoTag(value_.getLong(i));
      }
      size += dataSize;
      if (!getValueList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      valueMemoizedSerializedSize = dataSize;
    }
    if (java.lang.Double.doubleToRawLongBits(observedValue_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(12, observedValue_);
    }
    if (observedCount_ != 0L) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(13, observedCount_);
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
    if (!(obj instanceof org.tribuo.protos.core.CategoricalIDInfoProto)) {
      return super.equals(obj);
    }
    org.tribuo.protos.core.CategoricalIDInfoProto other = (org.tribuo.protos.core.CategoricalIDInfoProto) obj;

    if (!getName()
        .equals(other.getName())) return false;
    if (getCount()
        != other.getCount()) return false;
    if (getId()
        != other.getId()) return false;
    if (!getKeyList()
        .equals(other.getKeyList())) return false;
    if (!getValueList()
        .equals(other.getValueList())) return false;
    if (java.lang.Double.doubleToLongBits(getObservedValue())
        != java.lang.Double.doubleToLongBits(
            other.getObservedValue())) return false;
    if (getObservedCount()
        != other.getObservedCount()) return false;
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
    hash = (37 * hash) + NAME_FIELD_NUMBER;
    hash = (53 * hash) + getName().hashCode();
    hash = (37 * hash) + COUNT_FIELD_NUMBER;
    hash = (53 * hash) + getCount();
    hash = (37 * hash) + ID_FIELD_NUMBER;
    hash = (53 * hash) + getId();
    if (getKeyCount() > 0) {
      hash = (37 * hash) + KEY_FIELD_NUMBER;
      hash = (53 * hash) + getKeyList().hashCode();
    }
    if (getValueCount() > 0) {
      hash = (37 * hash) + VALUE_FIELD_NUMBER;
      hash = (53 * hash) + getValueList().hashCode();
    }
    hash = (37 * hash) + OBSERVED_VALUE_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getObservedValue()));
    hash = (37 * hash) + OBSERVED_COUNT_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        getObservedCount());
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.protos.core.CategoricalIDInfoProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.protos.core.CategoricalIDInfoProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.CategoricalIDInfoProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.protos.core.CategoricalIDInfoProto prototype) {
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
   *CategoricalIDInfo proto
   * </pre>
   *
   * Protobuf type {@code tribuo.core.CategoricalIDInfoProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.core.CategoricalIDInfoProto)
      org.tribuo.protos.core.CategoricalIDInfoProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_CategoricalIDInfoProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_CategoricalIDInfoProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.protos.core.CategoricalIDInfoProto.class, org.tribuo.protos.core.CategoricalIDInfoProto.Builder.class);
    }

    // Construct using org.tribuo.protos.core.CategoricalIDInfoProto.newBuilder()
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
      name_ = "";
      count_ = 0;
      id_ = 0;
      key_ = emptyDoubleList();
      value_ = emptyLongList();
      observedValue_ = 0D;
      observedCount_ = 0L;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_CategoricalIDInfoProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.protos.core.CategoricalIDInfoProto getDefaultInstanceForType() {
      return org.tribuo.protos.core.CategoricalIDInfoProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.protos.core.CategoricalIDInfoProto build() {
      org.tribuo.protos.core.CategoricalIDInfoProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.protos.core.CategoricalIDInfoProto buildPartial() {
      org.tribuo.protos.core.CategoricalIDInfoProto result = new org.tribuo.protos.core.CategoricalIDInfoProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.protos.core.CategoricalIDInfoProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.name_ = name_;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        result.count_ = count_;
      }
      if (((from_bitField0_ & 0x00000004) != 0)) {
        result.id_ = id_;
      }
      if (((from_bitField0_ & 0x00000008) != 0)) {
        key_.makeImmutable();
        result.key_ = key_;
      }
      if (((from_bitField0_ & 0x00000010) != 0)) {
        value_.makeImmutable();
        result.value_ = value_;
      }
      if (((from_bitField0_ & 0x00000020) != 0)) {
        result.observedValue_ = observedValue_;
      }
      if (((from_bitField0_ & 0x00000040) != 0)) {
        result.observedCount_ = observedCount_;
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
      if (other instanceof org.tribuo.protos.core.CategoricalIDInfoProto) {
        return mergeFrom((org.tribuo.protos.core.CategoricalIDInfoProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.protos.core.CategoricalIDInfoProto other) {
      if (other == org.tribuo.protos.core.CategoricalIDInfoProto.getDefaultInstance()) return this;
      if (!other.getName().isEmpty()) {
        name_ = other.name_;
        bitField0_ |= 0x00000001;
        onChanged();
      }
      if (other.getCount() != 0) {
        setCount(other.getCount());
      }
      if (other.getId() != 0) {
        setId(other.getId());
      }
      if (!other.key_.isEmpty()) {
        if (key_.isEmpty()) {
          key_ = other.key_;
          key_.makeImmutable();
          bitField0_ |= 0x00000008;
        } else {
          ensureKeyIsMutable();
          key_.addAll(other.key_);
        }
        onChanged();
      }
      if (!other.value_.isEmpty()) {
        if (value_.isEmpty()) {
          value_ = other.value_;
          value_.makeImmutable();
          bitField0_ |= 0x00000010;
        } else {
          ensureValueIsMutable();
          value_.addAll(other.value_);
        }
        onChanged();
      }
      if (other.getObservedValue() != 0D) {
        setObservedValue(other.getObservedValue());
      }
      if (other.getObservedCount() != 0L) {
        setObservedCount(other.getObservedCount());
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
              name_ = input.readStringRequireUtf8();
              bitField0_ |= 0x00000001;
              break;
            } // case 10
            case 16: {
              count_ = input.readInt32();
              bitField0_ |= 0x00000002;
              break;
            } // case 16
            case 24: {
              id_ = input.readInt32();
              bitField0_ |= 0x00000004;
              break;
            } // case 24
            case 81: {
              double v = input.readDouble();
              ensureKeyIsMutable();
              key_.addDouble(v);
              break;
            } // case 81
            case 82: {
              int length = input.readRawVarint32();
              int limit = input.pushLimit(length);
              int alloc = length > 4096 ? 4096 : length;
              ensureKeyIsMutable(alloc / 8);
              while (input.getBytesUntilLimit() > 0) {
                key_.addDouble(input.readDouble());
              }
              input.popLimit(limit);
              break;
            } // case 82
            case 88: {
              long v = input.readInt64();
              ensureValueIsMutable();
              value_.addLong(v);
              break;
            } // case 88
            case 90: {
              int length = input.readRawVarint32();
              int limit = input.pushLimit(length);
              ensureValueIsMutable();
              while (input.getBytesUntilLimit() > 0) {
                value_.addLong(input.readInt64());
              }
              input.popLimit(limit);
              break;
            } // case 90
            case 97: {
              observedValue_ = input.readDouble();
              bitField0_ |= 0x00000020;
              break;
            } // case 97
            case 104: {
              observedCount_ = input.readInt64();
              bitField0_ |= 0x00000040;
              break;
            } // case 104
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

    private java.lang.Object name_ = "";
    /**
     * <code>string name = 1;</code>
     * @return The name.
     */
    public java.lang.String getName() {
      java.lang.Object ref = name_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        name_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string name = 1;</code>
     * @return The bytes for name.
     */
    public com.google.protobuf.ByteString
        getNameBytes() {
      java.lang.Object ref = name_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        name_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string name = 1;</code>
     * @param value The name to set.
     * @return This builder for chaining.
     */
    public Builder setName(
        java.lang.String value) {
      if (value == null) { throw new NullPointerException(); }
      name_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>string name = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearName() {
      name_ = getDefaultInstance().getName();
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>string name = 1;</code>
     * @param value The bytes for name to set.
     * @return This builder for chaining.
     */
    public Builder setNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) { throw new NullPointerException(); }
      checkByteStringIsUtf8(value);
      name_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }

    private int count_ ;
    /**
     * <code>int32 count = 2;</code>
     * @return The count.
     */
    @java.lang.Override
    public int getCount() {
      return count_;
    }
    /**
     * <code>int32 count = 2;</code>
     * @param value The count to set.
     * @return This builder for chaining.
     */
    public Builder setCount(int value) {

      count_ = value;
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>int32 count = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearCount() {
      bitField0_ = (bitField0_ & ~0x00000002);
      count_ = 0;
      onChanged();
      return this;
    }

    private int id_ ;
    /**
     * <code>int32 id = 3;</code>
     * @return The id.
     */
    @java.lang.Override
    public int getId() {
      return id_;
    }
    /**
     * <code>int32 id = 3;</code>
     * @param value The id to set.
     * @return This builder for chaining.
     */
    public Builder setId(int value) {

      id_ = value;
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>int32 id = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearId() {
      bitField0_ = (bitField0_ & ~0x00000004);
      id_ = 0;
      onChanged();
      return this;
    }

    private com.google.protobuf.Internal.DoubleList key_ = emptyDoubleList();
    private void ensureKeyIsMutable() {
      if (!key_.isModifiable()) {
        key_ = makeMutableCopy(key_);
      }
      bitField0_ |= 0x00000008;
    }
    private void ensureKeyIsMutable(int capacity) {
      if (!key_.isModifiable()) {
        key_ = makeMutableCopy(key_, capacity);
      }
      bitField0_ |= 0x00000008;
    }
    /**
     * <code>repeated double key = 10;</code>
     * @return A list containing the key.
     */
    public java.util.List<java.lang.Double>
        getKeyList() {
      key_.makeImmutable();
      return key_;
    }
    /**
     * <code>repeated double key = 10;</code>
     * @return The count of key.
     */
    public int getKeyCount() {
      return key_.size();
    }
    /**
     * <code>repeated double key = 10;</code>
     * @param index The index of the element to return.
     * @return The key at the given index.
     */
    public double getKey(int index) {
      return key_.getDouble(index);
    }
    /**
     * <code>repeated double key = 10;</code>
     * @param index The index to set the value at.
     * @param value The key to set.
     * @return This builder for chaining.
     */
    public Builder setKey(
        int index, double value) {

      ensureKeyIsMutable();
      key_.setDouble(index, value);
      bitField0_ |= 0x00000008;
      onChanged();
      return this;
    }
    /**
     * <code>repeated double key = 10;</code>
     * @param value The key to add.
     * @return This builder for chaining.
     */
    public Builder addKey(double value) {

      ensureKeyIsMutable();
      key_.addDouble(value);
      bitField0_ |= 0x00000008;
      onChanged();
      return this;
    }
    /**
     * <code>repeated double key = 10;</code>
     * @param values The key to add.
     * @return This builder for chaining.
     */
    public Builder addAllKey(
        java.lang.Iterable<? extends java.lang.Double> values) {
      ensureKeyIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, key_);
      bitField0_ |= 0x00000008;
      onChanged();
      return this;
    }
    /**
     * <code>repeated double key = 10;</code>
     * @return This builder for chaining.
     */
    public Builder clearKey() {
      key_ = emptyDoubleList();
      bitField0_ = (bitField0_ & ~0x00000008);
      onChanged();
      return this;
    }

    private com.google.protobuf.Internal.LongList value_ = emptyLongList();
    private void ensureValueIsMutable() {
      if (!value_.isModifiable()) {
        value_ = makeMutableCopy(value_);
      }
      bitField0_ |= 0x00000010;
    }
    /**
     * <code>repeated int64 value = 11;</code>
     * @return A list containing the value.
     */
    public java.util.List<java.lang.Long>
        getValueList() {
      value_.makeImmutable();
      return value_;
    }
    /**
     * <code>repeated int64 value = 11;</code>
     * @return The count of value.
     */
    public int getValueCount() {
      return value_.size();
    }
    /**
     * <code>repeated int64 value = 11;</code>
     * @param index The index of the element to return.
     * @return The value at the given index.
     */
    public long getValue(int index) {
      return value_.getLong(index);
    }
    /**
     * <code>repeated int64 value = 11;</code>
     * @param index The index to set the value at.
     * @param value The value to set.
     * @return This builder for chaining.
     */
    public Builder setValue(
        int index, long value) {

      ensureValueIsMutable();
      value_.setLong(index, value);
      bitField0_ |= 0x00000010;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 value = 11;</code>
     * @param value The value to add.
     * @return This builder for chaining.
     */
    public Builder addValue(long value) {

      ensureValueIsMutable();
      value_.addLong(value);
      bitField0_ |= 0x00000010;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 value = 11;</code>
     * @param values The value to add.
     * @return This builder for chaining.
     */
    public Builder addAllValue(
        java.lang.Iterable<? extends java.lang.Long> values) {
      ensureValueIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, value_);
      bitField0_ |= 0x00000010;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 value = 11;</code>
     * @return This builder for chaining.
     */
    public Builder clearValue() {
      value_ = emptyLongList();
      bitField0_ = (bitField0_ & ~0x00000010);
      onChanged();
      return this;
    }

    private double observedValue_ ;
    /**
     * <code>double observed_value = 12;</code>
     * @return The observedValue.
     */
    @java.lang.Override
    public double getObservedValue() {
      return observedValue_;
    }
    /**
     * <code>double observed_value = 12;</code>
     * @param value The observedValue to set.
     * @return This builder for chaining.
     */
    public Builder setObservedValue(double value) {

      observedValue_ = value;
      bitField0_ |= 0x00000020;
      onChanged();
      return this;
    }
    /**
     * <code>double observed_value = 12;</code>
     * @return This builder for chaining.
     */
    public Builder clearObservedValue() {
      bitField0_ = (bitField0_ & ~0x00000020);
      observedValue_ = 0D;
      onChanged();
      return this;
    }

    private long observedCount_ ;
    /**
     * <code>int64 observed_count = 13;</code>
     * @return The observedCount.
     */
    @java.lang.Override
    public long getObservedCount() {
      return observedCount_;
    }
    /**
     * <code>int64 observed_count = 13;</code>
     * @param value The observedCount to set.
     * @return This builder for chaining.
     */
    public Builder setObservedCount(long value) {

      observedCount_ = value;
      bitField0_ |= 0x00000040;
      onChanged();
      return this;
    }
    /**
     * <code>int64 observed_count = 13;</code>
     * @return This builder for chaining.
     */
    public Builder clearObservedCount() {
      bitField0_ = (bitField0_ & ~0x00000040);
      observedCount_ = 0L;
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


    // @@protoc_insertion_point(builder_scope:tribuo.core.CategoricalIDInfoProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.core.CategoricalIDInfoProto)
  private static final org.tribuo.protos.core.CategoricalIDInfoProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.protos.core.CategoricalIDInfoProto();
  }

  public static org.tribuo.protos.core.CategoricalIDInfoProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<CategoricalIDInfoProto>
      PARSER = new com.google.protobuf.AbstractParser<CategoricalIDInfoProto>() {
    @java.lang.Override
    public CategoricalIDInfoProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<CategoricalIDInfoProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<CategoricalIDInfoProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.protos.core.CategoricalIDInfoProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


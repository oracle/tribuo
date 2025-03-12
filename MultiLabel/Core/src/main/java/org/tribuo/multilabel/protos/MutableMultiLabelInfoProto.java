// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-multilabel-core.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.multilabel.protos;

/**
 * <pre>
 *
 *MutableMultiLabelInfoProto
 * </pre>
 *
 * Protobuf type {@code tribuo.multilabel.MutableMultiLabelInfoProto}
 */
public final class MutableMultiLabelInfoProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.multilabel.MutableMultiLabelInfoProto)
    MutableMultiLabelInfoProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use MutableMultiLabelInfoProto.newBuilder() to construct.
  private MutableMultiLabelInfoProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private MutableMultiLabelInfoProto() {
    label_ =
        com.google.protobuf.LazyStringArrayList.emptyList();
    count_ = emptyLongList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new MutableMultiLabelInfoProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MutableMultiLabelInfoProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MutableMultiLabelInfoProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.multilabel.protos.MutableMultiLabelInfoProto.class, org.tribuo.multilabel.protos.MutableMultiLabelInfoProto.Builder.class);
  }

  public static final int LABEL_FIELD_NUMBER = 1;
  @SuppressWarnings("serial")
  private com.google.protobuf.LazyStringArrayList label_ =
      com.google.protobuf.LazyStringArrayList.emptyList();
  /**
   * <code>repeated string label = 1;</code>
   * @return A list containing the label.
   */
  public com.google.protobuf.ProtocolStringList
      getLabelList() {
    return label_;
  }
  /**
   * <code>repeated string label = 1;</code>
   * @return The count of label.
   */
  public int getLabelCount() {
    return label_.size();
  }
  /**
   * <code>repeated string label = 1;</code>
   * @param index The index of the element to return.
   * @return The label at the given index.
   */
  public java.lang.String getLabel(int index) {
    return label_.get(index);
  }
  /**
   * <code>repeated string label = 1;</code>
   * @param index The index of the value to return.
   * @return The bytes of the label at the given index.
   */
  public com.google.protobuf.ByteString
      getLabelBytes(int index) {
    return label_.getByteString(index);
  }

  public static final int COUNT_FIELD_NUMBER = 2;
  @SuppressWarnings("serial")
  private com.google.protobuf.Internal.LongList count_ =
      emptyLongList();
  /**
   * <code>repeated int64 count = 2;</code>
   * @return A list containing the count.
   */
  @java.lang.Override
  public java.util.List<java.lang.Long>
      getCountList() {
    return count_;
  }
  /**
   * <code>repeated int64 count = 2;</code>
   * @return The count of count.
   */
  public int getCountCount() {
    return count_.size();
  }
  /**
   * <code>repeated int64 count = 2;</code>
   * @param index The index of the element to return.
   * @return The count at the given index.
   */
  public long getCount(int index) {
    return count_.getLong(index);
  }
  private int countMemoizedSerializedSize = -1;

  public static final int UNKNOWNCOUNT_FIELD_NUMBER = 3;
  private int unknownCount_ = 0;
  /**
   * <code>int32 unknownCount = 3;</code>
   * @return The unknownCount.
   */
  @java.lang.Override
  public int getUnknownCount() {
    return unknownCount_;
  }

  public static final int TOTALCOUNT_FIELD_NUMBER = 4;
  private int totalCount_ = 0;
  /**
   * <code>int32 totalCount = 4;</code>
   * @return The totalCount.
   */
  @java.lang.Override
  public int getTotalCount() {
    return totalCount_;
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
    for (int i = 0; i < label_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, label_.getRaw(i));
    }
    if (getCountList().size() > 0) {
      output.writeUInt32NoTag(18);
      output.writeUInt32NoTag(countMemoizedSerializedSize);
    }
    for (int i = 0; i < count_.size(); i++) {
      output.writeInt64NoTag(count_.getLong(i));
    }
    if (unknownCount_ != 0) {
      output.writeInt32(3, unknownCount_);
    }
    if (totalCount_ != 0) {
      output.writeInt32(4, totalCount_);
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < label_.size(); i++) {
        dataSize += computeStringSizeNoTag(label_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getLabelList().size();
    }
    {
      int dataSize = 0;
      for (int i = 0; i < count_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeInt64SizeNoTag(count_.getLong(i));
      }
      size += dataSize;
      if (!getCountList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      countMemoizedSerializedSize = dataSize;
    }
    if (unknownCount_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(3, unknownCount_);
    }
    if (totalCount_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(4, totalCount_);
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
    if (!(obj instanceof org.tribuo.multilabel.protos.MutableMultiLabelInfoProto)) {
      return super.equals(obj);
    }
    org.tribuo.multilabel.protos.MutableMultiLabelInfoProto other = (org.tribuo.multilabel.protos.MutableMultiLabelInfoProto) obj;

    if (!getLabelList()
        .equals(other.getLabelList())) return false;
    if (!getCountList()
        .equals(other.getCountList())) return false;
    if (getUnknownCount()
        != other.getUnknownCount()) return false;
    if (getTotalCount()
        != other.getTotalCount()) return false;
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
    if (getLabelCount() > 0) {
      hash = (37 * hash) + LABEL_FIELD_NUMBER;
      hash = (53 * hash) + getLabelList().hashCode();
    }
    if (getCountCount() > 0) {
      hash = (37 * hash) + COUNT_FIELD_NUMBER;
      hash = (53 * hash) + getCountList().hashCode();
    }
    hash = (37 * hash) + UNKNOWNCOUNT_FIELD_NUMBER;
    hash = (53 * hash) + getUnknownCount();
    hash = (37 * hash) + TOTALCOUNT_FIELD_NUMBER;
    hash = (53 * hash) + getTotalCount();
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.multilabel.protos.MutableMultiLabelInfoProto prototype) {
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
   *MutableMultiLabelInfoProto
   * </pre>
   *
   * Protobuf type {@code tribuo.multilabel.MutableMultiLabelInfoProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.multilabel.MutableMultiLabelInfoProto)
      org.tribuo.multilabel.protos.MutableMultiLabelInfoProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MutableMultiLabelInfoProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MutableMultiLabelInfoProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.multilabel.protos.MutableMultiLabelInfoProto.class, org.tribuo.multilabel.protos.MutableMultiLabelInfoProto.Builder.class);
    }

    // Construct using org.tribuo.multilabel.protos.MutableMultiLabelInfoProto.newBuilder()
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
      label_ =
          com.google.protobuf.LazyStringArrayList.emptyList();
      count_ = emptyLongList();
      unknownCount_ = 0;
      totalCount_ = 0;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MutableMultiLabelInfoProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.MutableMultiLabelInfoProto getDefaultInstanceForType() {
      return org.tribuo.multilabel.protos.MutableMultiLabelInfoProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.MutableMultiLabelInfoProto build() {
      org.tribuo.multilabel.protos.MutableMultiLabelInfoProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.MutableMultiLabelInfoProto buildPartial() {
      org.tribuo.multilabel.protos.MutableMultiLabelInfoProto result = new org.tribuo.multilabel.protos.MutableMultiLabelInfoProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.multilabel.protos.MutableMultiLabelInfoProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        label_.makeImmutable();
        result.label_ = label_;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        count_.makeImmutable();
        result.count_ = count_;
      }
      if (((from_bitField0_ & 0x00000004) != 0)) {
        result.unknownCount_ = unknownCount_;
      }
      if (((from_bitField0_ & 0x00000008) != 0)) {
        result.totalCount_ = totalCount_;
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
      if (other instanceof org.tribuo.multilabel.protos.MutableMultiLabelInfoProto) {
        return mergeFrom((org.tribuo.multilabel.protos.MutableMultiLabelInfoProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.multilabel.protos.MutableMultiLabelInfoProto other) {
      if (other == org.tribuo.multilabel.protos.MutableMultiLabelInfoProto.getDefaultInstance()) return this;
      if (!other.label_.isEmpty()) {
        if (label_.isEmpty()) {
          label_ = other.label_;
          bitField0_ |= 0x00000001;
        } else {
          ensureLabelIsMutable();
          label_.addAll(other.label_);
        }
        onChanged();
      }
      if (!other.count_.isEmpty()) {
        if (count_.isEmpty()) {
          count_ = other.count_;
          count_.makeImmutable();
          bitField0_ |= 0x00000002;
        } else {
          ensureCountIsMutable();
          count_.addAll(other.count_);
        }
        onChanged();
      }
      if (other.getUnknownCount() != 0) {
        setUnknownCount(other.getUnknownCount());
      }
      if (other.getTotalCount() != 0) {
        setTotalCount(other.getTotalCount());
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
              java.lang.String s = input.readStringRequireUtf8();
              ensureLabelIsMutable();
              label_.add(s);
              break;
            } // case 10
            case 16: {
              long v = input.readInt64();
              ensureCountIsMutable();
              count_.addLong(v);
              break;
            } // case 16
            case 18: {
              int length = input.readRawVarint32();
              int limit = input.pushLimit(length);
              ensureCountIsMutable();
              while (input.getBytesUntilLimit() > 0) {
                count_.addLong(input.readInt64());
              }
              input.popLimit(limit);
              break;
            } // case 18
            case 24: {
              unknownCount_ = input.readInt32();
              bitField0_ |= 0x00000004;
              break;
            } // case 24
            case 32: {
              totalCount_ = input.readInt32();
              bitField0_ |= 0x00000008;
              break;
            } // case 32
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

    private com.google.protobuf.LazyStringArrayList label_ =
        com.google.protobuf.LazyStringArrayList.emptyList();
    private void ensureLabelIsMutable() {
      if (!label_.isModifiable()) {
        label_ = new com.google.protobuf.LazyStringArrayList(label_);
      }
      bitField0_ |= 0x00000001;
    }
    /**
     * <code>repeated string label = 1;</code>
     * @return A list containing the label.
     */
    public com.google.protobuf.ProtocolStringList
        getLabelList() {
      label_.makeImmutable();
      return label_;
    }
    /**
     * <code>repeated string label = 1;</code>
     * @return The count of label.
     */
    public int getLabelCount() {
      return label_.size();
    }
    /**
     * <code>repeated string label = 1;</code>
     * @param index The index of the element to return.
     * @return The label at the given index.
     */
    public java.lang.String getLabel(int index) {
      return label_.get(index);
    }
    /**
     * <code>repeated string label = 1;</code>
     * @param index The index of the value to return.
     * @return The bytes of the label at the given index.
     */
    public com.google.protobuf.ByteString
        getLabelBytes(int index) {
      return label_.getByteString(index);
    }
    /**
     * <code>repeated string label = 1;</code>
     * @param index The index to set the value at.
     * @param value The label to set.
     * @return This builder for chaining.
     */
    public Builder setLabel(
        int index, java.lang.String value) {
      if (value == null) { throw new NullPointerException(); }
      ensureLabelIsMutable();
      label_.set(index, value);
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>repeated string label = 1;</code>
     * @param value The label to add.
     * @return This builder for chaining.
     */
    public Builder addLabel(
        java.lang.String value) {
      if (value == null) { throw new NullPointerException(); }
      ensureLabelIsMutable();
      label_.add(value);
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>repeated string label = 1;</code>
     * @param values The label to add.
     * @return This builder for chaining.
     */
    public Builder addAllLabel(
        java.lang.Iterable<java.lang.String> values) {
      ensureLabelIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, label_);
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>repeated string label = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearLabel() {
      label_ =
        com.google.protobuf.LazyStringArrayList.emptyList();
      bitField0_ = (bitField0_ & ~0x00000001);;
      onChanged();
      return this;
    }
    /**
     * <code>repeated string label = 1;</code>
     * @param value The bytes of the label to add.
     * @return This builder for chaining.
     */
    public Builder addLabelBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) { throw new NullPointerException(); }
      checkByteStringIsUtf8(value);
      ensureLabelIsMutable();
      label_.add(value);
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }

    private com.google.protobuf.Internal.LongList count_ = emptyLongList();
    private void ensureCountIsMutable() {
      if (!count_.isModifiable()) {
        count_ = makeMutableCopy(count_);
      }
      bitField0_ |= 0x00000002;
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @return A list containing the count.
     */
    public java.util.List<java.lang.Long>
        getCountList() {
      count_.makeImmutable();
      return count_;
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @return The count of count.
     */
    public int getCountCount() {
      return count_.size();
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @param index The index of the element to return.
     * @return The count at the given index.
     */
    public long getCount(int index) {
      return count_.getLong(index);
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @param index The index to set the value at.
     * @param value The count to set.
     * @return This builder for chaining.
     */
    public Builder setCount(
        int index, long value) {

      ensureCountIsMutable();
      count_.setLong(index, value);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @param value The count to add.
     * @return This builder for chaining.
     */
    public Builder addCount(long value) {

      ensureCountIsMutable();
      count_.addLong(value);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @param values The count to add.
     * @return This builder for chaining.
     */
    public Builder addAllCount(
        java.lang.Iterable<? extends java.lang.Long> values) {
      ensureCountIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, count_);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearCount() {
      count_ = emptyLongList();
      bitField0_ = (bitField0_ & ~0x00000002);
      onChanged();
      return this;
    }

    private int unknownCount_ ;
    /**
     * <code>int32 unknownCount = 3;</code>
     * @return The unknownCount.
     */
    @java.lang.Override
    public int getUnknownCount() {
      return unknownCount_;
    }
    /**
     * <code>int32 unknownCount = 3;</code>
     * @param value The unknownCount to set.
     * @return This builder for chaining.
     */
    public Builder setUnknownCount(int value) {

      unknownCount_ = value;
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>int32 unknownCount = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearUnknownCount() {
      bitField0_ = (bitField0_ & ~0x00000004);
      unknownCount_ = 0;
      onChanged();
      return this;
    }

    private int totalCount_ ;
    /**
     * <code>int32 totalCount = 4;</code>
     * @return The totalCount.
     */
    @java.lang.Override
    public int getTotalCount() {
      return totalCount_;
    }
    /**
     * <code>int32 totalCount = 4;</code>
     * @param value The totalCount to set.
     * @return This builder for chaining.
     */
    public Builder setTotalCount(int value) {

      totalCount_ = value;
      bitField0_ |= 0x00000008;
      onChanged();
      return this;
    }
    /**
     * <code>int32 totalCount = 4;</code>
     * @return This builder for chaining.
     */
    public Builder clearTotalCount() {
      bitField0_ = (bitField0_ & ~0x00000008);
      totalCount_ = 0;
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


    // @@protoc_insertion_point(builder_scope:tribuo.multilabel.MutableMultiLabelInfoProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.multilabel.MutableMultiLabelInfoProto)
  private static final org.tribuo.multilabel.protos.MutableMultiLabelInfoProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.multilabel.protos.MutableMultiLabelInfoProto();
  }

  public static org.tribuo.multilabel.protos.MutableMultiLabelInfoProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<MutableMultiLabelInfoProto>
      PARSER = new com.google.protobuf.AbstractParser<MutableMultiLabelInfoProto>() {
    @java.lang.Override
    public MutableMultiLabelInfoProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<MutableMultiLabelInfoProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<MutableMultiLabelInfoProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.multilabel.protos.MutableMultiLabelInfoProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


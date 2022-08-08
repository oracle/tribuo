// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-multilabel-core.proto

package org.tribuo.multilabel.protos;

/**
 * <pre>
 *ImmutableMultiLabelInfoProto
 * </pre>
 *
 * Protobuf type {@code tribuo.multilabel.ImmutableMultiLabelInfoProto}
 */
public final class ImmutableMultiLabelInfoProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.multilabel.ImmutableMultiLabelInfoProto)
    ImmutableMultiLabelInfoProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use ImmutableMultiLabelInfoProto.newBuilder() to construct.
  private ImmutableMultiLabelInfoProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ImmutableMultiLabelInfoProto() {
    label_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    count_ = emptyLongList();
    id_ = emptyIntList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new ImmutableMultiLabelInfoProto();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ImmutableMultiLabelInfoProto(
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
          case 10: {
            java.lang.String s = input.readStringRequireUtf8();
            if (!((mutable_bitField0_ & 0x00000001) != 0)) {
              label_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000001;
            }
            label_.add(s);
            break;
          }
          case 16: {
            if (!((mutable_bitField0_ & 0x00000002) != 0)) {
              count_ = newLongList();
              mutable_bitField0_ |= 0x00000002;
            }
            count_.addLong(input.readInt64());
            break;
          }
          case 18: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            if (!((mutable_bitField0_ & 0x00000002) != 0) && input.getBytesUntilLimit() > 0) {
              count_ = newLongList();
              mutable_bitField0_ |= 0x00000002;
            }
            while (input.getBytesUntilLimit() > 0) {
              count_.addLong(input.readInt64());
            }
            input.popLimit(limit);
            break;
          }
          case 24: {
            if (!((mutable_bitField0_ & 0x00000004) != 0)) {
              id_ = newIntList();
              mutable_bitField0_ |= 0x00000004;
            }
            id_.addInt(input.readInt32());
            break;
          }
          case 26: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            if (!((mutable_bitField0_ & 0x00000004) != 0) && input.getBytesUntilLimit() > 0) {
              id_ = newIntList();
              mutable_bitField0_ |= 0x00000004;
            }
            while (input.getBytesUntilLimit() > 0) {
              id_.addInt(input.readInt32());
            }
            input.popLimit(limit);
            break;
          }
          case 32: {

            unknownCount_ = input.readInt32();
            break;
          }
          case 40: {

            totalCount_ = input.readInt32();
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
        label_ = label_.getUnmodifiableView();
      }
      if (((mutable_bitField0_ & 0x00000002) != 0)) {
        count_.makeImmutable(); // C
      }
      if (((mutable_bitField0_ & 0x00000004) != 0)) {
        id_.makeImmutable(); // C
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_ImmutableMultiLabelInfoProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_ImmutableMultiLabelInfoProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto.class, org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto.Builder.class);
  }

  public static final int LABEL_FIELD_NUMBER = 1;
  private com.google.protobuf.LazyStringList label_;
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
  private com.google.protobuf.Internal.LongList count_;
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

  public static final int ID_FIELD_NUMBER = 3;
  private com.google.protobuf.Internal.IntList id_;
  /**
   * <code>repeated int32 id = 3;</code>
   * @return A list containing the id.
   */
  @java.lang.Override
  public java.util.List<java.lang.Integer>
      getIdList() {
    return id_;
  }
  /**
   * <code>repeated int32 id = 3;</code>
   * @return The count of id.
   */
  public int getIdCount() {
    return id_.size();
  }
  /**
   * <code>repeated int32 id = 3;</code>
   * @param index The index of the element to return.
   * @return The id at the given index.
   */
  public int getId(int index) {
    return id_.getInt(index);
  }
  private int idMemoizedSerializedSize = -1;

  public static final int UNKNOWNCOUNT_FIELD_NUMBER = 4;
  private int unknownCount_;
  /**
   * <code>int32 unknownCount = 4;</code>
   * @return The unknownCount.
   */
  @java.lang.Override
  public int getUnknownCount() {
    return unknownCount_;
  }

  public static final int TOTALCOUNT_FIELD_NUMBER = 5;
  private int totalCount_;
  /**
   * <code>int32 totalCount = 5;</code>
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
    if (getIdList().size() > 0) {
      output.writeUInt32NoTag(26);
      output.writeUInt32NoTag(idMemoizedSerializedSize);
    }
    for (int i = 0; i < id_.size(); i++) {
      output.writeInt32NoTag(id_.getInt(i));
    }
    if (unknownCount_ != 0) {
      output.writeInt32(4, unknownCount_);
    }
    if (totalCount_ != 0) {
      output.writeInt32(5, totalCount_);
    }
    unknownFields.writeTo(output);
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
    {
      int dataSize = 0;
      for (int i = 0; i < id_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeInt32SizeNoTag(id_.getInt(i));
      }
      size += dataSize;
      if (!getIdList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      idMemoizedSerializedSize = dataSize;
    }
    if (unknownCount_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(4, unknownCount_);
    }
    if (totalCount_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(5, totalCount_);
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
    if (!(obj instanceof org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto)) {
      return super.equals(obj);
    }
    org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto other = (org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto) obj;

    if (!getLabelList()
        .equals(other.getLabelList())) return false;
    if (!getCountList()
        .equals(other.getCountList())) return false;
    if (!getIdList()
        .equals(other.getIdList())) return false;
    if (getUnknownCount()
        != other.getUnknownCount()) return false;
    if (getTotalCount()
        != other.getTotalCount()) return false;
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
    if (getLabelCount() > 0) {
      hash = (37 * hash) + LABEL_FIELD_NUMBER;
      hash = (53 * hash) + getLabelList().hashCode();
    }
    if (getCountCount() > 0) {
      hash = (37 * hash) + COUNT_FIELD_NUMBER;
      hash = (53 * hash) + getCountList().hashCode();
    }
    if (getIdCount() > 0) {
      hash = (37 * hash) + ID_FIELD_NUMBER;
      hash = (53 * hash) + getIdList().hashCode();
    }
    hash = (37 * hash) + UNKNOWNCOUNT_FIELD_NUMBER;
    hash = (53 * hash) + getUnknownCount();
    hash = (37 * hash) + TOTALCOUNT_FIELD_NUMBER;
    hash = (53 * hash) + getTotalCount();
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto prototype) {
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
   *ImmutableMultiLabelInfoProto
   * </pre>
   *
   * Protobuf type {@code tribuo.multilabel.ImmutableMultiLabelInfoProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.multilabel.ImmutableMultiLabelInfoProto)
      org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_ImmutableMultiLabelInfoProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_ImmutableMultiLabelInfoProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto.class, org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto.Builder.class);
    }

    // Construct using org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto.newBuilder()
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
      label_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      count_ = emptyLongList();
      bitField0_ = (bitField0_ & ~0x00000002);
      id_ = emptyIntList();
      bitField0_ = (bitField0_ & ~0x00000004);
      unknownCount_ = 0;

      totalCount_ = 0;

      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_ImmutableMultiLabelInfoProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto getDefaultInstanceForType() {
      return org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto build() {
      org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto buildPartial() {
      org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto result = new org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto(this);
      int from_bitField0_ = bitField0_;
      if (((bitField0_ & 0x00000001) != 0)) {
        label_ = label_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.label_ = label_;
      if (((bitField0_ & 0x00000002) != 0)) {
        count_.makeImmutable();
        bitField0_ = (bitField0_ & ~0x00000002);
      }
      result.count_ = count_;
      if (((bitField0_ & 0x00000004) != 0)) {
        id_.makeImmutable();
        bitField0_ = (bitField0_ & ~0x00000004);
      }
      result.id_ = id_;
      result.unknownCount_ = unknownCount_;
      result.totalCount_ = totalCount_;
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
      if (other instanceof org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto) {
        return mergeFrom((org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto other) {
      if (other == org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto.getDefaultInstance()) return this;
      if (!other.label_.isEmpty()) {
        if (label_.isEmpty()) {
          label_ = other.label_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureLabelIsMutable();
          label_.addAll(other.label_);
        }
        onChanged();
      }
      if (!other.count_.isEmpty()) {
        if (count_.isEmpty()) {
          count_ = other.count_;
          bitField0_ = (bitField0_ & ~0x00000002);
        } else {
          ensureCountIsMutable();
          count_.addAll(other.count_);
        }
        onChanged();
      }
      if (!other.id_.isEmpty()) {
        if (id_.isEmpty()) {
          id_ = other.id_;
          bitField0_ = (bitField0_ & ~0x00000004);
        } else {
          ensureIdIsMutable();
          id_.addAll(other.id_);
        }
        onChanged();
      }
      if (other.getUnknownCount() != 0) {
        setUnknownCount(other.getUnknownCount());
      }
      if (other.getTotalCount() != 0) {
        setTotalCount(other.getTotalCount());
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
      org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private com.google.protobuf.LazyStringList label_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureLabelIsMutable() {
      if (!((bitField0_ & 0x00000001) != 0)) {
        label_ = new com.google.protobuf.LazyStringArrayList(label_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated string label = 1;</code>
     * @return A list containing the label.
     */
    public com.google.protobuf.ProtocolStringList
        getLabelList() {
      return label_.getUnmodifiableView();
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
      if (value == null) {
    throw new NullPointerException();
  }
  ensureLabelIsMutable();
      label_.set(index, value);
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
      if (value == null) {
    throw new NullPointerException();
  }
  ensureLabelIsMutable();
      label_.add(value);
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
      onChanged();
      return this;
    }
    /**
     * <code>repeated string label = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearLabel() {
      label_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
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
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      ensureLabelIsMutable();
      label_.add(value);
      onChanged();
      return this;
    }

    private com.google.protobuf.Internal.LongList count_ = emptyLongList();
    private void ensureCountIsMutable() {
      if (!((bitField0_ & 0x00000002) != 0)) {
        count_ = mutableCopy(count_);
        bitField0_ |= 0x00000002;
       }
    }
    /**
     * <code>repeated int64 count = 2;</code>
     * @return A list containing the count.
     */
    public java.util.List<java.lang.Long>
        getCountList() {
      return ((bitField0_ & 0x00000002) != 0) ?
               java.util.Collections.unmodifiableList(count_) : count_;
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

    private com.google.protobuf.Internal.IntList id_ = emptyIntList();
    private void ensureIdIsMutable() {
      if (!((bitField0_ & 0x00000004) != 0)) {
        id_ = mutableCopy(id_);
        bitField0_ |= 0x00000004;
       }
    }
    /**
     * <code>repeated int32 id = 3;</code>
     * @return A list containing the id.
     */
    public java.util.List<java.lang.Integer>
        getIdList() {
      return ((bitField0_ & 0x00000004) != 0) ?
               java.util.Collections.unmodifiableList(id_) : id_;
    }
    /**
     * <code>repeated int32 id = 3;</code>
     * @return The count of id.
     */
    public int getIdCount() {
      return id_.size();
    }
    /**
     * <code>repeated int32 id = 3;</code>
     * @param index The index of the element to return.
     * @return The id at the given index.
     */
    public int getId(int index) {
      return id_.getInt(index);
    }
    /**
     * <code>repeated int32 id = 3;</code>
     * @param index The index to set the value at.
     * @param value The id to set.
     * @return This builder for chaining.
     */
    public Builder setId(
        int index, int value) {
      ensureIdIsMutable();
      id_.setInt(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 id = 3;</code>
     * @param value The id to add.
     * @return This builder for chaining.
     */
    public Builder addId(int value) {
      ensureIdIsMutable();
      id_.addInt(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 id = 3;</code>
     * @param values The id to add.
     * @return This builder for chaining.
     */
    public Builder addAllId(
        java.lang.Iterable<? extends java.lang.Integer> values) {
      ensureIdIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, id_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 id = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearId() {
      id_ = emptyIntList();
      bitField0_ = (bitField0_ & ~0x00000004);
      onChanged();
      return this;
    }

    private int unknownCount_ ;
    /**
     * <code>int32 unknownCount = 4;</code>
     * @return The unknownCount.
     */
    @java.lang.Override
    public int getUnknownCount() {
      return unknownCount_;
    }
    /**
     * <code>int32 unknownCount = 4;</code>
     * @param value The unknownCount to set.
     * @return This builder for chaining.
     */
    public Builder setUnknownCount(int value) {
      
      unknownCount_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 unknownCount = 4;</code>
     * @return This builder for chaining.
     */
    public Builder clearUnknownCount() {
      
      unknownCount_ = 0;
      onChanged();
      return this;
    }

    private int totalCount_ ;
    /**
     * <code>int32 totalCount = 5;</code>
     * @return The totalCount.
     */
    @java.lang.Override
    public int getTotalCount() {
      return totalCount_;
    }
    /**
     * <code>int32 totalCount = 5;</code>
     * @param value The totalCount to set.
     * @return This builder for chaining.
     */
    public Builder setTotalCount(int value) {
      
      totalCount_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 totalCount = 5;</code>
     * @return This builder for chaining.
     */
    public Builder clearTotalCount() {
      
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


    // @@protoc_insertion_point(builder_scope:tribuo.multilabel.ImmutableMultiLabelInfoProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.multilabel.ImmutableMultiLabelInfoProto)
  private static final org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto();
  }

  public static org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<ImmutableMultiLabelInfoProto>
      PARSER = new com.google.protobuf.AbstractParser<ImmutableMultiLabelInfoProto>() {
    @java.lang.Override
    public ImmutableMultiLabelInfoProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new ImmutableMultiLabelInfoProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ImmutableMultiLabelInfoProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ImmutableMultiLabelInfoProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.multilabel.protos.ImmutableMultiLabelInfoProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-tensorflow.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.interop.tensorflow.protos;

/**
 * <pre>
 *
 *TensorFlowUtil.TensorTuple proto
 * </pre>
 *
 * Protobuf type {@code tribuo.interop.tensorflow.TensorTupleProto}
 */
public final class TensorTupleProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.interop.tensorflow.TensorTupleProto)
    TensorTupleProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use TensorTupleProto.newBuilder() to construct.
  private TensorTupleProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private TensorTupleProto() {
    className_ = "";
    shape_ = emptyLongList();
    data_ = com.google.protobuf.ByteString.EMPTY;
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new TensorTupleProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.interop.tensorflow.protos.TribuoTensorflow.internal_static_tribuo_interop_tensorflow_TensorTupleProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.interop.tensorflow.protos.TribuoTensorflow.internal_static_tribuo_interop_tensorflow_TensorTupleProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.interop.tensorflow.protos.TensorTupleProto.class, org.tribuo.interop.tensorflow.protos.TensorTupleProto.Builder.class);
  }

  public static final int CLASS_NAME_FIELD_NUMBER = 1;
  @SuppressWarnings("serial")
  private volatile java.lang.Object className_ = "";
  /**
   * <code>string class_name = 1;</code>
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
   * <code>string class_name = 1;</code>
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

  public static final int SHAPE_FIELD_NUMBER = 2;
  @SuppressWarnings("serial")
  private com.google.protobuf.Internal.LongList shape_ =
      emptyLongList();
  /**
   * <code>repeated int64 shape = 2;</code>
   * @return A list containing the shape.
   */
  @java.lang.Override
  public java.util.List<java.lang.Long>
      getShapeList() {
    return shape_;
  }
  /**
   * <code>repeated int64 shape = 2;</code>
   * @return The count of shape.
   */
  public int getShapeCount() {
    return shape_.size();
  }
  /**
   * <code>repeated int64 shape = 2;</code>
   * @param index The index of the element to return.
   * @return The shape at the given index.
   */
  public long getShape(int index) {
    return shape_.getLong(index);
  }
  private int shapeMemoizedSerializedSize = -1;

  public static final int DATA_FIELD_NUMBER = 3;
  private com.google.protobuf.ByteString data_ = com.google.protobuf.ByteString.EMPTY;
  /**
   * <code>bytes data = 3;</code>
   * @return The data.
   */
  @java.lang.Override
  public com.google.protobuf.ByteString getData() {
    return data_;
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
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(className_)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, className_);
    }
    if (getShapeList().size() > 0) {
      output.writeUInt32NoTag(18);
      output.writeUInt32NoTag(shapeMemoizedSerializedSize);
    }
    for (int i = 0; i < shape_.size(); i++) {
      output.writeInt64NoTag(shape_.getLong(i));
    }
    if (!data_.isEmpty()) {
      output.writeBytes(3, data_);
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(className_)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, className_);
    }
    {
      int dataSize = 0;
      for (int i = 0; i < shape_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeInt64SizeNoTag(shape_.getLong(i));
      }
      size += dataSize;
      if (!getShapeList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      shapeMemoizedSerializedSize = dataSize;
    }
    if (!data_.isEmpty()) {
      size += com.google.protobuf.CodedOutputStream
        .computeBytesSize(3, data_);
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
    if (!(obj instanceof org.tribuo.interop.tensorflow.protos.TensorTupleProto)) {
      return super.equals(obj);
    }
    org.tribuo.interop.tensorflow.protos.TensorTupleProto other = (org.tribuo.interop.tensorflow.protos.TensorTupleProto) obj;

    if (!getClassName()
        .equals(other.getClassName())) return false;
    if (!getShapeList()
        .equals(other.getShapeList())) return false;
    if (!getData()
        .equals(other.getData())) return false;
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
    hash = (37 * hash) + CLASS_NAME_FIELD_NUMBER;
    hash = (53 * hash) + getClassName().hashCode();
    if (getShapeCount() > 0) {
      hash = (37 * hash) + SHAPE_FIELD_NUMBER;
      hash = (53 * hash) + getShapeList().hashCode();
    }
    hash = (37 * hash) + DATA_FIELD_NUMBER;
    hash = (53 * hash) + getData().hashCode();
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.interop.tensorflow.protos.TensorTupleProto prototype) {
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
   *TensorFlowUtil.TensorTuple proto
   * </pre>
   *
   * Protobuf type {@code tribuo.interop.tensorflow.TensorTupleProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.interop.tensorflow.TensorTupleProto)
      org.tribuo.interop.tensorflow.protos.TensorTupleProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.interop.tensorflow.protos.TribuoTensorflow.internal_static_tribuo_interop_tensorflow_TensorTupleProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.interop.tensorflow.protos.TribuoTensorflow.internal_static_tribuo_interop_tensorflow_TensorTupleProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.interop.tensorflow.protos.TensorTupleProto.class, org.tribuo.interop.tensorflow.protos.TensorTupleProto.Builder.class);
    }

    // Construct using org.tribuo.interop.tensorflow.protos.TensorTupleProto.newBuilder()
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
      className_ = "";
      shape_ = emptyLongList();
      data_ = com.google.protobuf.ByteString.EMPTY;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.interop.tensorflow.protos.TribuoTensorflow.internal_static_tribuo_interop_tensorflow_TensorTupleProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.interop.tensorflow.protos.TensorTupleProto getDefaultInstanceForType() {
      return org.tribuo.interop.tensorflow.protos.TensorTupleProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.interop.tensorflow.protos.TensorTupleProto build() {
      org.tribuo.interop.tensorflow.protos.TensorTupleProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.interop.tensorflow.protos.TensorTupleProto buildPartial() {
      org.tribuo.interop.tensorflow.protos.TensorTupleProto result = new org.tribuo.interop.tensorflow.protos.TensorTupleProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.interop.tensorflow.protos.TensorTupleProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.className_ = className_;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        shape_.makeImmutable();
        result.shape_ = shape_;
      }
      if (((from_bitField0_ & 0x00000004) != 0)) {
        result.data_ = data_;
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
      if (other instanceof org.tribuo.interop.tensorflow.protos.TensorTupleProto) {
        return mergeFrom((org.tribuo.interop.tensorflow.protos.TensorTupleProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.interop.tensorflow.protos.TensorTupleProto other) {
      if (other == org.tribuo.interop.tensorflow.protos.TensorTupleProto.getDefaultInstance()) return this;
      if (!other.getClassName().isEmpty()) {
        className_ = other.className_;
        bitField0_ |= 0x00000001;
        onChanged();
      }
      if (!other.shape_.isEmpty()) {
        if (shape_.isEmpty()) {
          shape_ = other.shape_;
          shape_.makeImmutable();
          bitField0_ |= 0x00000002;
        } else {
          ensureShapeIsMutable();
          shape_.addAll(other.shape_);
        }
        onChanged();
      }
      if (other.getData() != com.google.protobuf.ByteString.EMPTY) {
        setData(other.getData());
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
              className_ = input.readStringRequireUtf8();
              bitField0_ |= 0x00000001;
              break;
            } // case 10
            case 16: {
              long v = input.readInt64();
              ensureShapeIsMutable();
              shape_.addLong(v);
              break;
            } // case 16
            case 18: {
              int length = input.readRawVarint32();
              int limit = input.pushLimit(length);
              ensureShapeIsMutable();
              while (input.getBytesUntilLimit() > 0) {
                shape_.addLong(input.readInt64());
              }
              input.popLimit(limit);
              break;
            } // case 18
            case 26: {
              data_ = input.readBytes();
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

    private java.lang.Object className_ = "";
    /**
     * <code>string class_name = 1;</code>
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
     * <code>string class_name = 1;</code>
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
     * <code>string class_name = 1;</code>
     * @param value The className to set.
     * @return This builder for chaining.
     */
    public Builder setClassName(
        java.lang.String value) {
      if (value == null) { throw new NullPointerException(); }
      className_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>string class_name = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearClassName() {
      className_ = getDefaultInstance().getClassName();
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>string class_name = 1;</code>
     * @param value The bytes for className to set.
     * @return This builder for chaining.
     */
    public Builder setClassNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) { throw new NullPointerException(); }
      checkByteStringIsUtf8(value);
      className_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }

    private com.google.protobuf.Internal.LongList shape_ = emptyLongList();
    private void ensureShapeIsMutable() {
      if (!shape_.isModifiable()) {
        shape_ = makeMutableCopy(shape_);
      }
      bitField0_ |= 0x00000002;
    }
    /**
     * <code>repeated int64 shape = 2;</code>
     * @return A list containing the shape.
     */
    public java.util.List<java.lang.Long>
        getShapeList() {
      shape_.makeImmutable();
      return shape_;
    }
    /**
     * <code>repeated int64 shape = 2;</code>
     * @return The count of shape.
     */
    public int getShapeCount() {
      return shape_.size();
    }
    /**
     * <code>repeated int64 shape = 2;</code>
     * @param index The index of the element to return.
     * @return The shape at the given index.
     */
    public long getShape(int index) {
      return shape_.getLong(index);
    }
    /**
     * <code>repeated int64 shape = 2;</code>
     * @param index The index to set the value at.
     * @param value The shape to set.
     * @return This builder for chaining.
     */
    public Builder setShape(
        int index, long value) {

      ensureShapeIsMutable();
      shape_.setLong(index, value);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 shape = 2;</code>
     * @param value The shape to add.
     * @return This builder for chaining.
     */
    public Builder addShape(long value) {

      ensureShapeIsMutable();
      shape_.addLong(value);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 shape = 2;</code>
     * @param values The shape to add.
     * @return This builder for chaining.
     */
    public Builder addAllShape(
        java.lang.Iterable<? extends java.lang.Long> values) {
      ensureShapeIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, shape_);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 shape = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearShape() {
      shape_ = emptyLongList();
      bitField0_ = (bitField0_ & ~0x00000002);
      onChanged();
      return this;
    }

    private com.google.protobuf.ByteString data_ = com.google.protobuf.ByteString.EMPTY;
    /**
     * <code>bytes data = 3;</code>
     * @return The data.
     */
    @java.lang.Override
    public com.google.protobuf.ByteString getData() {
      return data_;
    }
    /**
     * <code>bytes data = 3;</code>
     * @param value The data to set.
     * @return This builder for chaining.
     */
    public Builder setData(com.google.protobuf.ByteString value) {
      if (value == null) { throw new NullPointerException(); }
      data_ = value;
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>bytes data = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearData() {
      bitField0_ = (bitField0_ & ~0x00000004);
      data_ = getDefaultInstance().getData();
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


    // @@protoc_insertion_point(builder_scope:tribuo.interop.tensorflow.TensorTupleProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.interop.tensorflow.TensorTupleProto)
  private static final org.tribuo.interop.tensorflow.protos.TensorTupleProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.interop.tensorflow.protos.TensorTupleProto();
  }

  public static org.tribuo.interop.tensorflow.protos.TensorTupleProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<TensorTupleProto>
      PARSER = new com.google.protobuf.AbstractParser<TensorTupleProto>() {
    @java.lang.Override
    public TensorTupleProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<TensorTupleProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<TensorTupleProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.interop.tensorflow.protos.TensorTupleProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


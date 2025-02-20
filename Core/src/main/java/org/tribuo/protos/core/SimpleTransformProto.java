// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.protos.core;

/**
 * <pre>
 *
 *SimpleTransform proto
 * </pre>
 *
 * Protobuf type {@code tribuo.core.SimpleTransformProto}
 */
public final class SimpleTransformProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.core.SimpleTransformProto)
    SimpleTransformProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use SimpleTransformProto.newBuilder() to construct.
  private SimpleTransformProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private SimpleTransformProto() {
    op_ = "";
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new SimpleTransformProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_SimpleTransformProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_SimpleTransformProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.protos.core.SimpleTransformProto.class, org.tribuo.protos.core.SimpleTransformProto.Builder.class);
  }

  public static final int OP_FIELD_NUMBER = 1;
  @SuppressWarnings("serial")
  private volatile java.lang.Object op_ = "";
  /**
   * <code>string op = 1;</code>
   * @return The op.
   */
  @java.lang.Override
  public java.lang.String getOp() {
    java.lang.Object ref = op_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      op_ = s;
      return s;
    }
  }
  /**
   * <code>string op = 1;</code>
   * @return The bytes for op.
   */
  @java.lang.Override
  public com.google.protobuf.ByteString
      getOpBytes() {
    java.lang.Object ref = op_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      op_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int FIRST_OPERAND_FIELD_NUMBER = 2;
  private double firstOperand_ = 0D;
  /**
   * <code>double first_operand = 2;</code>
   * @return The firstOperand.
   */
  @java.lang.Override
  public double getFirstOperand() {
    return firstOperand_;
  }

  public static final int SECOND_OPERAND_FIELD_NUMBER = 3;
  private double secondOperand_ = 0D;
  /**
   * <code>double second_operand = 3;</code>
   * @return The secondOperand.
   */
  @java.lang.Override
  public double getSecondOperand() {
    return secondOperand_;
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
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(op_)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, op_);
    }
    if (java.lang.Double.doubleToRawLongBits(firstOperand_) != 0) {
      output.writeDouble(2, firstOperand_);
    }
    if (java.lang.Double.doubleToRawLongBits(secondOperand_) != 0) {
      output.writeDouble(3, secondOperand_);
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(op_)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, op_);
    }
    if (java.lang.Double.doubleToRawLongBits(firstOperand_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(2, firstOperand_);
    }
    if (java.lang.Double.doubleToRawLongBits(secondOperand_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(3, secondOperand_);
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
    if (!(obj instanceof org.tribuo.protos.core.SimpleTransformProto)) {
      return super.equals(obj);
    }
    org.tribuo.protos.core.SimpleTransformProto other = (org.tribuo.protos.core.SimpleTransformProto) obj;

    if (!getOp()
        .equals(other.getOp())) return false;
    if (java.lang.Double.doubleToLongBits(getFirstOperand())
        != java.lang.Double.doubleToLongBits(
            other.getFirstOperand())) return false;
    if (java.lang.Double.doubleToLongBits(getSecondOperand())
        != java.lang.Double.doubleToLongBits(
            other.getSecondOperand())) return false;
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
    hash = (37 * hash) + OP_FIELD_NUMBER;
    hash = (53 * hash) + getOp().hashCode();
    hash = (37 * hash) + FIRST_OPERAND_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getFirstOperand()));
    hash = (37 * hash) + SECOND_OPERAND_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getSecondOperand()));
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.protos.core.SimpleTransformProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.protos.core.SimpleTransformProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.SimpleTransformProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.protos.core.SimpleTransformProto prototype) {
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
   *SimpleTransform proto
   * </pre>
   *
   * Protobuf type {@code tribuo.core.SimpleTransformProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.core.SimpleTransformProto)
      org.tribuo.protos.core.SimpleTransformProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_SimpleTransformProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_SimpleTransformProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.protos.core.SimpleTransformProto.class, org.tribuo.protos.core.SimpleTransformProto.Builder.class);
    }

    // Construct using org.tribuo.protos.core.SimpleTransformProto.newBuilder()
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
      op_ = "";
      firstOperand_ = 0D;
      secondOperand_ = 0D;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_SimpleTransformProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.protos.core.SimpleTransformProto getDefaultInstanceForType() {
      return org.tribuo.protos.core.SimpleTransformProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.protos.core.SimpleTransformProto build() {
      org.tribuo.protos.core.SimpleTransformProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.protos.core.SimpleTransformProto buildPartial() {
      org.tribuo.protos.core.SimpleTransformProto result = new org.tribuo.protos.core.SimpleTransformProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.protos.core.SimpleTransformProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.op_ = op_;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        result.firstOperand_ = firstOperand_;
      }
      if (((from_bitField0_ & 0x00000004) != 0)) {
        result.secondOperand_ = secondOperand_;
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
      if (other instanceof org.tribuo.protos.core.SimpleTransformProto) {
        return mergeFrom((org.tribuo.protos.core.SimpleTransformProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.protos.core.SimpleTransformProto other) {
      if (other == org.tribuo.protos.core.SimpleTransformProto.getDefaultInstance()) return this;
      if (!other.getOp().isEmpty()) {
        op_ = other.op_;
        bitField0_ |= 0x00000001;
        onChanged();
      }
      if (other.getFirstOperand() != 0D) {
        setFirstOperand(other.getFirstOperand());
      }
      if (other.getSecondOperand() != 0D) {
        setSecondOperand(other.getSecondOperand());
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
              op_ = input.readStringRequireUtf8();
              bitField0_ |= 0x00000001;
              break;
            } // case 10
            case 17: {
              firstOperand_ = input.readDouble();
              bitField0_ |= 0x00000002;
              break;
            } // case 17
            case 25: {
              secondOperand_ = input.readDouble();
              bitField0_ |= 0x00000004;
              break;
            } // case 25
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

    private java.lang.Object op_ = "";
    /**
     * <code>string op = 1;</code>
     * @return The op.
     */
    public java.lang.String getOp() {
      java.lang.Object ref = op_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        op_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string op = 1;</code>
     * @return The bytes for op.
     */
    public com.google.protobuf.ByteString
        getOpBytes() {
      java.lang.Object ref = op_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        op_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string op = 1;</code>
     * @param value The op to set.
     * @return This builder for chaining.
     */
    public Builder setOp(
        java.lang.String value) {
      if (value == null) { throw new NullPointerException(); }
      op_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>string op = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearOp() {
      op_ = getDefaultInstance().getOp();
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>string op = 1;</code>
     * @param value The bytes for op to set.
     * @return This builder for chaining.
     */
    public Builder setOpBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) { throw new NullPointerException(); }
      checkByteStringIsUtf8(value);
      op_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }

    private double firstOperand_ ;
    /**
     * <code>double first_operand = 2;</code>
     * @return The firstOperand.
     */
    @java.lang.Override
    public double getFirstOperand() {
      return firstOperand_;
    }
    /**
     * <code>double first_operand = 2;</code>
     * @param value The firstOperand to set.
     * @return This builder for chaining.
     */
    public Builder setFirstOperand(double value) {

      firstOperand_ = value;
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>double first_operand = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearFirstOperand() {
      bitField0_ = (bitField0_ & ~0x00000002);
      firstOperand_ = 0D;
      onChanged();
      return this;
    }

    private double secondOperand_ ;
    /**
     * <code>double second_operand = 3;</code>
     * @return The secondOperand.
     */
    @java.lang.Override
    public double getSecondOperand() {
      return secondOperand_;
    }
    /**
     * <code>double second_operand = 3;</code>
     * @param value The secondOperand to set.
     * @return This builder for chaining.
     */
    public Builder setSecondOperand(double value) {

      secondOperand_ = value;
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>double second_operand = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearSecondOperand() {
      bitField0_ = (bitField0_ & ~0x00000004);
      secondOperand_ = 0D;
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


    // @@protoc_insertion_point(builder_scope:tribuo.core.SimpleTransformProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.core.SimpleTransformProto)
  private static final org.tribuo.protos.core.SimpleTransformProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.protos.core.SimpleTransformProto();
  }

  public static org.tribuo.protos.core.SimpleTransformProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<SimpleTransformProto>
      PARSER = new com.google.protobuf.AbstractParser<SimpleTransformProto>() {
    @java.lang.Override
    public SimpleTransformProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<SimpleTransformProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<SimpleTransformProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.protos.core.SimpleTransformProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


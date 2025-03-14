// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-multilabel-core.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.multilabel.protos;

/**
 * <pre>
 *
 *MultiLabel proto
 * </pre>
 *
 * Protobuf type {@code tribuo.multilabel.MultiLabelProto}
 */
public final class MultiLabelProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.multilabel.MultiLabelProto)
    MultiLabelProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use MultiLabelProto.newBuilder() to construct.
  private MultiLabelProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private MultiLabelProto() {
    label_ =
        com.google.protobuf.LazyStringArrayList.emptyList();
    lblScore_ = emptyDoubleList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new MultiLabelProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MultiLabelProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MultiLabelProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.multilabel.protos.MultiLabelProto.class, org.tribuo.multilabel.protos.MultiLabelProto.Builder.class);
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

  public static final int LBLSCORE_FIELD_NUMBER = 2;
  @SuppressWarnings("serial")
  private com.google.protobuf.Internal.DoubleList lblScore_ =
      emptyDoubleList();
  /**
   * <code>repeated double lblScore = 2;</code>
   * @return A list containing the lblScore.
   */
  @java.lang.Override
  public java.util.List<java.lang.Double>
      getLblScoreList() {
    return lblScore_;
  }
  /**
   * <code>repeated double lblScore = 2;</code>
   * @return The count of lblScore.
   */
  public int getLblScoreCount() {
    return lblScore_.size();
  }
  /**
   * <code>repeated double lblScore = 2;</code>
   * @param index The index of the element to return.
   * @return The lblScore at the given index.
   */
  public double getLblScore(int index) {
    return lblScore_.getDouble(index);
  }
  private int lblScoreMemoizedSerializedSize = -1;

  public static final int OVERALLSCORE_FIELD_NUMBER = 3;
  private double overallScore_ = 0D;
  /**
   * <code>double overallScore = 3;</code>
   * @return The overallScore.
   */
  @java.lang.Override
  public double getOverallScore() {
    return overallScore_;
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
    if (getLblScoreList().size() > 0) {
      output.writeUInt32NoTag(18);
      output.writeUInt32NoTag(lblScoreMemoizedSerializedSize);
    }
    for (int i = 0; i < lblScore_.size(); i++) {
      output.writeDoubleNoTag(lblScore_.getDouble(i));
    }
    if (java.lang.Double.doubleToRawLongBits(overallScore_) != 0) {
      output.writeDouble(3, overallScore_);
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
      dataSize = 8 * getLblScoreList().size();
      size += dataSize;
      if (!getLblScoreList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      lblScoreMemoizedSerializedSize = dataSize;
    }
    if (java.lang.Double.doubleToRawLongBits(overallScore_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(3, overallScore_);
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
    if (!(obj instanceof org.tribuo.multilabel.protos.MultiLabelProto)) {
      return super.equals(obj);
    }
    org.tribuo.multilabel.protos.MultiLabelProto other = (org.tribuo.multilabel.protos.MultiLabelProto) obj;

    if (!getLabelList()
        .equals(other.getLabelList())) return false;
    if (!getLblScoreList()
        .equals(other.getLblScoreList())) return false;
    if (java.lang.Double.doubleToLongBits(getOverallScore())
        != java.lang.Double.doubleToLongBits(
            other.getOverallScore())) return false;
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
    if (getLblScoreCount() > 0) {
      hash = (37 * hash) + LBLSCORE_FIELD_NUMBER;
      hash = (53 * hash) + getLblScoreList().hashCode();
    }
    hash = (37 * hash) + OVERALLSCORE_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getOverallScore()));
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.multilabel.protos.MultiLabelProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.multilabel.protos.MultiLabelProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.multilabel.protos.MultiLabelProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.multilabel.protos.MultiLabelProto prototype) {
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
   *MultiLabel proto
   * </pre>
   *
   * Protobuf type {@code tribuo.multilabel.MultiLabelProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.multilabel.MultiLabelProto)
      org.tribuo.multilabel.protos.MultiLabelProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MultiLabelProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MultiLabelProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.multilabel.protos.MultiLabelProto.class, org.tribuo.multilabel.protos.MultiLabelProto.Builder.class);
    }

    // Construct using org.tribuo.multilabel.protos.MultiLabelProto.newBuilder()
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
      lblScore_ = emptyDoubleList();
      overallScore_ = 0D;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.multilabel.protos.TribuoMultilabelCore.internal_static_tribuo_multilabel_MultiLabelProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.MultiLabelProto getDefaultInstanceForType() {
      return org.tribuo.multilabel.protos.MultiLabelProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.MultiLabelProto build() {
      org.tribuo.multilabel.protos.MultiLabelProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.multilabel.protos.MultiLabelProto buildPartial() {
      org.tribuo.multilabel.protos.MultiLabelProto result = new org.tribuo.multilabel.protos.MultiLabelProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.multilabel.protos.MultiLabelProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        label_.makeImmutable();
        result.label_ = label_;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        lblScore_.makeImmutable();
        result.lblScore_ = lblScore_;
      }
      if (((from_bitField0_ & 0x00000004) != 0)) {
        result.overallScore_ = overallScore_;
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
      if (other instanceof org.tribuo.multilabel.protos.MultiLabelProto) {
        return mergeFrom((org.tribuo.multilabel.protos.MultiLabelProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.multilabel.protos.MultiLabelProto other) {
      if (other == org.tribuo.multilabel.protos.MultiLabelProto.getDefaultInstance()) return this;
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
      if (!other.lblScore_.isEmpty()) {
        if (lblScore_.isEmpty()) {
          lblScore_ = other.lblScore_;
          lblScore_.makeImmutable();
          bitField0_ |= 0x00000002;
        } else {
          ensureLblScoreIsMutable();
          lblScore_.addAll(other.lblScore_);
        }
        onChanged();
      }
      if (other.getOverallScore() != 0D) {
        setOverallScore(other.getOverallScore());
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
            case 17: {
              double v = input.readDouble();
              ensureLblScoreIsMutable();
              lblScore_.addDouble(v);
              break;
            } // case 17
            case 18: {
              int length = input.readRawVarint32();
              int limit = input.pushLimit(length);
              int alloc = length > 4096 ? 4096 : length;
              ensureLblScoreIsMutable(alloc / 8);
              while (input.getBytesUntilLimit() > 0) {
                lblScore_.addDouble(input.readDouble());
              }
              input.popLimit(limit);
              break;
            } // case 18
            case 25: {
              overallScore_ = input.readDouble();
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

    private com.google.protobuf.Internal.DoubleList lblScore_ = emptyDoubleList();
    private void ensureLblScoreIsMutable() {
      if (!lblScore_.isModifiable()) {
        lblScore_ = makeMutableCopy(lblScore_);
      }
      bitField0_ |= 0x00000002;
    }
    private void ensureLblScoreIsMutable(int capacity) {
      if (!lblScore_.isModifiable()) {
        lblScore_ = makeMutableCopy(lblScore_, capacity);
      }
      bitField0_ |= 0x00000002;
    }
    /**
     * <code>repeated double lblScore = 2;</code>
     * @return A list containing the lblScore.
     */
    public java.util.List<java.lang.Double>
        getLblScoreList() {
      lblScore_.makeImmutable();
      return lblScore_;
    }
    /**
     * <code>repeated double lblScore = 2;</code>
     * @return The count of lblScore.
     */
    public int getLblScoreCount() {
      return lblScore_.size();
    }
    /**
     * <code>repeated double lblScore = 2;</code>
     * @param index The index of the element to return.
     * @return The lblScore at the given index.
     */
    public double getLblScore(int index) {
      return lblScore_.getDouble(index);
    }
    /**
     * <code>repeated double lblScore = 2;</code>
     * @param index The index to set the value at.
     * @param value The lblScore to set.
     * @return This builder for chaining.
     */
    public Builder setLblScore(
        int index, double value) {

      ensureLblScoreIsMutable();
      lblScore_.setDouble(index, value);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated double lblScore = 2;</code>
     * @param value The lblScore to add.
     * @return This builder for chaining.
     */
    public Builder addLblScore(double value) {

      ensureLblScoreIsMutable();
      lblScore_.addDouble(value);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated double lblScore = 2;</code>
     * @param values The lblScore to add.
     * @return This builder for chaining.
     */
    public Builder addAllLblScore(
        java.lang.Iterable<? extends java.lang.Double> values) {
      ensureLblScoreIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, lblScore_);
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>repeated double lblScore = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearLblScore() {
      lblScore_ = emptyDoubleList();
      bitField0_ = (bitField0_ & ~0x00000002);
      onChanged();
      return this;
    }

    private double overallScore_ ;
    /**
     * <code>double overallScore = 3;</code>
     * @return The overallScore.
     */
    @java.lang.Override
    public double getOverallScore() {
      return overallScore_;
    }
    /**
     * <code>double overallScore = 3;</code>
     * @param value The overallScore to set.
     * @return This builder for chaining.
     */
    public Builder setOverallScore(double value) {

      overallScore_ = value;
      bitField0_ |= 0x00000004;
      onChanged();
      return this;
    }
    /**
     * <code>double overallScore = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearOverallScore() {
      bitField0_ = (bitField0_ & ~0x00000004);
      overallScore_ = 0D;
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


    // @@protoc_insertion_point(builder_scope:tribuo.multilabel.MultiLabelProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.multilabel.MultiLabelProto)
  private static final org.tribuo.multilabel.protos.MultiLabelProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.multilabel.protos.MultiLabelProto();
  }

  public static org.tribuo.multilabel.protos.MultiLabelProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<MultiLabelProto>
      PARSER = new com.google.protobuf.AbstractParser<MultiLabelProto>() {
    @java.lang.Override
    public MultiLabelProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<MultiLabelProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<MultiLabelProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.multilabel.protos.MultiLabelProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


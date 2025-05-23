// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-anomaly-core.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.anomaly.protos;

/**
 * <pre>
 *
 *Event proto
 * </pre>
 *
 * Protobuf type {@code tribuo.anomaly.EventProto}
 */
public final class EventProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.anomaly.EventProto)
    EventProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use EventProto.newBuilder() to construct.
  private EventProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private EventProto() {
    event_ = 0;
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new EventProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.anomaly.protos.TribuoAnomalyCore.internal_static_tribuo_anomaly_EventProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.anomaly.protos.TribuoAnomalyCore.internal_static_tribuo_anomaly_EventProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.anomaly.protos.EventProto.class, org.tribuo.anomaly.protos.EventProto.Builder.class);
  }

  /**
   * Protobuf enum {@code tribuo.anomaly.EventProto.EventType}
   */
  public enum EventType
      implements com.google.protobuf.ProtocolMessageEnum {
    /**
     * <code>EXPECTED = 0;</code>
     */
    EXPECTED(0),
    /**
     * <code>ANOMALOUS = 1;</code>
     */
    ANOMALOUS(1),
    /**
     * <code>UNKNOWN = -1;</code>
     */
    UNKNOWN(-1),
    UNRECOGNIZED(-1),
    ;

    /**
     * <code>EXPECTED = 0;</code>
     */
    public static final int EXPECTED_VALUE = 0;
    /**
     * <code>ANOMALOUS = 1;</code>
     */
    public static final int ANOMALOUS_VALUE = 1;
    /**
     * <code>UNKNOWN = -1;</code>
     */
    public static final int UNKNOWN_VALUE = -1;


    public final int getNumber() {
      if (this == UNRECOGNIZED) {
        throw new java.lang.IllegalArgumentException(
            "Can't get the number of an unknown enum value.");
      }
      return value;
    }

    /**
     * @param value The numeric wire value of the corresponding enum entry.
     * @return The enum associated with the given numeric wire value.
     * @deprecated Use {@link #forNumber(int)} instead.
     */
    @java.lang.Deprecated
    public static EventType valueOf(int value) {
      return forNumber(value);
    }

    /**
     * @param value The numeric wire value of the corresponding enum entry.
     * @return The enum associated with the given numeric wire value.
     */
    public static EventType forNumber(int value) {
      switch (value) {
        case 0: return EXPECTED;
        case 1: return ANOMALOUS;
        case -1: return UNKNOWN;
        default: return null;
      }
    }

    public static com.google.protobuf.Internal.EnumLiteMap<EventType>
        internalGetValueMap() {
      return internalValueMap;
    }
    private static final com.google.protobuf.Internal.EnumLiteMap<
        EventType> internalValueMap =
          new com.google.protobuf.Internal.EnumLiteMap<EventType>() {
            public EventType findValueByNumber(int number) {
              return EventType.forNumber(number);
            }
          };

    public final com.google.protobuf.Descriptors.EnumValueDescriptor
        getValueDescriptor() {
      if (this == UNRECOGNIZED) {
        throw new java.lang.IllegalStateException(
            "Can't get the descriptor of an unrecognized enum value.");
      }
      return getDescriptor().getValues().get(ordinal());
    }
    public final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptorForType() {
      return getDescriptor();
    }
    public static final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptor() {
      return org.tribuo.anomaly.protos.EventProto.getDescriptor().getEnumTypes().get(0);
    }

    private static final EventType[] VALUES = values();

    public static EventType valueOf(
        com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
      if (desc.getType() != getDescriptor()) {
        throw new java.lang.IllegalArgumentException(
          "EnumValueDescriptor is not for this type.");
      }
      if (desc.getIndex() == -1) {
        return UNRECOGNIZED;
      }
      return VALUES[desc.getIndex()];
    }

    private final int value;

    private EventType(int value) {
      this.value = value;
    }

    // @@protoc_insertion_point(enum_scope:tribuo.anomaly.EventProto.EventType)
  }

  public static final int EVENT_FIELD_NUMBER = 1;
  private int event_ = 0;
  /**
   * <code>.tribuo.anomaly.EventProto.EventType event = 1;</code>
   * @return The enum numeric value on the wire for event.
   */
  @java.lang.Override public int getEventValue() {
    return event_;
  }
  /**
   * <code>.tribuo.anomaly.EventProto.EventType event = 1;</code>
   * @return The event.
   */
  @java.lang.Override public org.tribuo.anomaly.protos.EventProto.EventType getEvent() {
    org.tribuo.anomaly.protos.EventProto.EventType result = org.tribuo.anomaly.protos.EventProto.EventType.forNumber(event_);
    return result == null ? org.tribuo.anomaly.protos.EventProto.EventType.UNRECOGNIZED : result;
  }

  public static final int SCORE_FIELD_NUMBER = 2;
  private double score_ = 0D;
  /**
   * <code>double score = 2;</code>
   * @return The score.
   */
  @java.lang.Override
  public double getScore() {
    return score_;
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
    if (event_ != org.tribuo.anomaly.protos.EventProto.EventType.EXPECTED.getNumber()) {
      output.writeEnum(1, event_);
    }
    if (java.lang.Double.doubleToRawLongBits(score_) != 0) {
      output.writeDouble(2, score_);
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (event_ != org.tribuo.anomaly.protos.EventProto.EventType.EXPECTED.getNumber()) {
      size += com.google.protobuf.CodedOutputStream
        .computeEnumSize(1, event_);
    }
    if (java.lang.Double.doubleToRawLongBits(score_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(2, score_);
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
    if (!(obj instanceof org.tribuo.anomaly.protos.EventProto)) {
      return super.equals(obj);
    }
    org.tribuo.anomaly.protos.EventProto other = (org.tribuo.anomaly.protos.EventProto) obj;

    if (event_ != other.event_) return false;
    if (java.lang.Double.doubleToLongBits(getScore())
        != java.lang.Double.doubleToLongBits(
            other.getScore())) return false;
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
    hash = (37 * hash) + EVENT_FIELD_NUMBER;
    hash = (53 * hash) + event_;
    hash = (37 * hash) + SCORE_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getScore()));
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.anomaly.protos.EventProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.anomaly.protos.EventProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.anomaly.protos.EventProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.anomaly.protos.EventProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.anomaly.protos.EventProto prototype) {
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
   *Event proto
   * </pre>
   *
   * Protobuf type {@code tribuo.anomaly.EventProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.anomaly.EventProto)
      org.tribuo.anomaly.protos.EventProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.anomaly.protos.TribuoAnomalyCore.internal_static_tribuo_anomaly_EventProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.anomaly.protos.TribuoAnomalyCore.internal_static_tribuo_anomaly_EventProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.anomaly.protos.EventProto.class, org.tribuo.anomaly.protos.EventProto.Builder.class);
    }

    // Construct using org.tribuo.anomaly.protos.EventProto.newBuilder()
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
      event_ = 0;
      score_ = 0D;
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.anomaly.protos.TribuoAnomalyCore.internal_static_tribuo_anomaly_EventProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.anomaly.protos.EventProto getDefaultInstanceForType() {
      return org.tribuo.anomaly.protos.EventProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.anomaly.protos.EventProto build() {
      org.tribuo.anomaly.protos.EventProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.anomaly.protos.EventProto buildPartial() {
      org.tribuo.anomaly.protos.EventProto result = new org.tribuo.anomaly.protos.EventProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.anomaly.protos.EventProto result) {
      int from_bitField0_ = bitField0_;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.event_ = event_;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        result.score_ = score_;
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
      if (other instanceof org.tribuo.anomaly.protos.EventProto) {
        return mergeFrom((org.tribuo.anomaly.protos.EventProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.anomaly.protos.EventProto other) {
      if (other == org.tribuo.anomaly.protos.EventProto.getDefaultInstance()) return this;
      if (other.event_ != 0) {
        setEventValue(other.getEventValue());
      }
      if (other.getScore() != 0D) {
        setScore(other.getScore());
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
              event_ = input.readEnum();
              bitField0_ |= 0x00000001;
              break;
            } // case 8
            case 17: {
              score_ = input.readDouble();
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

    private int event_ = 0;
    /**
     * <code>.tribuo.anomaly.EventProto.EventType event = 1;</code>
     * @return The enum numeric value on the wire for event.
     */
    @java.lang.Override public int getEventValue() {
      return event_;
    }
    /**
     * <code>.tribuo.anomaly.EventProto.EventType event = 1;</code>
     * @param value The enum numeric value on the wire for event to set.
     * @return This builder for chaining.
     */
    public Builder setEventValue(int value) {
      event_ = value;
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.anomaly.EventProto.EventType event = 1;</code>
     * @return The event.
     */
    @java.lang.Override
    public org.tribuo.anomaly.protos.EventProto.EventType getEvent() {
      org.tribuo.anomaly.protos.EventProto.EventType result = org.tribuo.anomaly.protos.EventProto.EventType.forNumber(event_);
      return result == null ? org.tribuo.anomaly.protos.EventProto.EventType.UNRECOGNIZED : result;
    }
    /**
     * <code>.tribuo.anomaly.EventProto.EventType event = 1;</code>
     * @param value The event to set.
     * @return This builder for chaining.
     */
    public Builder setEvent(org.tribuo.anomaly.protos.EventProto.EventType value) {
      if (value == null) {
        throw new NullPointerException();
      }
      bitField0_ |= 0x00000001;
      event_ = value.getNumber();
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.anomaly.EventProto.EventType event = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearEvent() {
      bitField0_ = (bitField0_ & ~0x00000001);
      event_ = 0;
      onChanged();
      return this;
    }

    private double score_ ;
    /**
     * <code>double score = 2;</code>
     * @return The score.
     */
    @java.lang.Override
    public double getScore() {
      return score_;
    }
    /**
     * <code>double score = 2;</code>
     * @param value The score to set.
     * @return This builder for chaining.
     */
    public Builder setScore(double value) {

      score_ = value;
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>double score = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearScore() {
      bitField0_ = (bitField0_ & ~0x00000002);
      score_ = 0D;
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


    // @@protoc_insertion_point(builder_scope:tribuo.anomaly.EventProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.anomaly.EventProto)
  private static final org.tribuo.anomaly.protos.EventProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.anomaly.protos.EventProto();
  }

  public static org.tribuo.anomaly.protos.EventProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<EventProto>
      PARSER = new com.google.protobuf.AbstractParser<EventProto>() {
    @java.lang.Override
    public EventProto parsePartialFrom(
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

  public static com.google.protobuf.Parser<EventProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<EventProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.anomaly.protos.EventProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}


package org.tribuo;

import static java.lang.annotation.ElementType.TYPE;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(TYPE)
public @interface ProtobufClass {
    Class<? extends com.google.protobuf.GeneratedMessageV3> serializedClass();
    Class<? extends com.google.protobuf.GeneratedMessageV3> serializedData();
}

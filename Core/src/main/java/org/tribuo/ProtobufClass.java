package org.tribuo;

import static java.lang.annotation.ElementType.TYPE;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import com.google.protobuf.GeneratedMessageV3;

@Retention(RetentionPolicy.RUNTIME)
@Target(TYPE)
public @interface ProtobufClass {
    Class<? extends GeneratedMessageV3> serializedClass();
    Class<? extends GeneratedMessageV3> serializedData() default GeneratedMessageV3.class;
    int version() default 0;
}

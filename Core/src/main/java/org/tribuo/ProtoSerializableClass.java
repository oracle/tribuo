package org.tribuo;

import static java.lang.annotation.ElementType.TYPE;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import com.google.protobuf.GeneratedMessageV3;
import com.google.protobuf.Message;

@Retention(RetentionPolicy.RUNTIME)
@Target(TYPE)
public @interface ProtoSerializableClass {
    Class<? extends Message> serializedDataClass() default GeneratedMessageV3.class;
    int version() default 0;
}

package org.tribuo;

import static java.lang.annotation.ElementType.TYPE;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import com.google.protobuf.GeneratedMessageV3;
import com.google.protobuf.Message;

/**
 * Mark a class as being {@link ProtoSerializable} and specify
 * the class type used to serialize the "serialized_data"
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(TYPE)
public @interface ProtoSerializableClass {
    /**
     * Specifies
     * @return
     */
    Class<? extends Message> serializedDataClass() default GeneratedMessageV3.class;
    int version() default 0;
}

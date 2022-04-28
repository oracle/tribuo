package org.tribuo.protos;

import static java.lang.annotation.ElementType.FIELD;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(FIELD)
public @interface ProtoSerializableArrayField {
    
    public static final String DEFAULT_FIELD_NAME = "[DEFAULT_FIELD_NAME]";

    int sinceVersion() default 0;
    String name() default DEFAULT_FIELD_NAME; 
}

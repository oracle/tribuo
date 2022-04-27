package org.tribuo.protos;

import static java.lang.annotation.ElementType.FIELD;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(FIELD)
public @interface ProtoSerializableMapValuesField {
    int sinceVersion() default 0;
    String valuesName();
}

package org.tribuo.protos;

import static java.lang.annotation.ElementType.FIELD;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(FIELD)
public @interface ProtoSerializableKeysValuesField {
    int sinceVersion() default 0;
    String keysName();
    String valuesName();
}

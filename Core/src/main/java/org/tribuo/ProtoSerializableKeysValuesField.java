package org.tribuo;

import static java.lang.annotation.ElementType.FIELD;

import java.lang.annotation.Repeatable;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(FIELD)
public @interface ProtoSerializableKeysValuesField {
    int sinceVersion() default 0;
    String keyName();
    String valueName();
}

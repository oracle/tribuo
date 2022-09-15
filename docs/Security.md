# Security Considerations

Tribuo is a library designed to be incorporated into larger programs.
Therefore, we consider the trust boundary to live somewhere outside of Tribuo
in the larger program. While we provide data loaders, we expect them to be used
on trusted or cleaned data. We expect that the larger program will control
access to Tribuo's interfaces. Tribuo's data loaders and other interfaces
perform defensive copying and other standard procedures when user code crosses
over into Tribuo's internals. For performance reasons, however, this defensive
behavior is not generally the case for calls within Tribuo (e.g., the linear
algebra library exposes mutable state to reduce copying).

## Serialized files
Tribuo models are stored as Java serialized objects. Due to the inherent issues
with Java serialization, these object files should only be loaded and saved to
trusted locations where third parties do not have access. We have provided a
[JEP 290](https://openjdk.java.net/jeps/290) [filter](jep-290-filter.txt)
which will allow the deserialization of only the classes found in the Tribuo
library. This filter should be enabled on the code paths which deserialize
models or datasets. As Tribuo supports Java 8+, and JEP 290 is an addition to
the Java 8 API from 8u121, the best way to use the filter for the main 
programs provided with Tribuo is by setting it as a process-wide flag.  
Additionally, when running with a security manager, Tribuo will need access to
the relevant filesystem locations to load or save model files. See the section 
on [Configuration](#Configuration) for more details.

In Tribuo 4.3 we introduced protobuf based serialization for all supported Java
serializable types. This is the preferred serialization mechanism, and Java
serialization support will be removed in the next major release of Tribuo.

## Database access
Tribuo provides a SQL interface that can load data via a JDBC connection. As
it's frequently necessary to load data via a joined query from an unknown
schema, Tribuo *does not* validate the input SQL. It is expected that the
program developer will perform this validation since they know the schema from
which they are loading. Tribuo supports connections via public key wallets via
JDBC. To use this functionality, supply the wallet configuration to the JVM as
a system property and use the constructors that accept a java.util.Properties
instance with the appropriate configuration.

## Native code
Tribuo uses several native libraries via JNI interfaces. Native code has
different considerations as compared to pure Java code because native code can
cause issues in the running JVM. We are active contributors to all the native
ML libraries that Tribuo uses, and we fix issues upstream if we find them.
Nevertheless, you should think carefully before running a model that requires
native code inside an application container like a JavaEE or JakartaEE server.
Multiple instances of Tribuo running inside separate containers may cause
issues with JNI library loading due to ClassLoader security considerations.

## SecurityManager configuration
Tribuo uses [OLCUT](https://github.com/oracle/olcut)'s configuration and
provenance systems, which use reflection to construct and inspect classes.
Therefore, when running with a Java security manager, you need to give the
OLCUT jar appropriate permissions. We have tested this set of permissions,
which allows the configuration and provenance systems to work:

    // OLCUT permissions
    grant codeBase "file:/path/to/olcut/olcut-core-5.2.0.jar" {
            permission java.lang.RuntimePermission "accessDeclaredMembers";
            permission java.lang.reflect.ReflectPermission "suppressAccessChecks";
            permission java.util.logging.LoggingPermission "control";
            permission java.io.FilePermission "<<ALL FILES>>", "read";
            permission java.util.PropertyPermission "*", "read,write";
    };

The read FilePermission can be restricted to the jars which contain
configuration files, configuration files on disk, and the locations of
serialised objects. The FilePermission in this example provides access to the
complete filesystem because the necessary read locations are program specific.
This scope should be narrowed based on your requirements. If you need to save
an OLCUT configuration, you will also need to add write permissions for the
save location.

Tribuo uses `ForkJoinPool` for parallelism, which requires the `modifyThread`
and `modifyThreadGroup` privileges when running under a `java.lang.SecurityManager`.
Therefore classes which have parallel execution inside will require those
permissions in addition to the ones listed for OLCUT above.

File read and write permissions are necessary for Tribuo to be able to
load and save models; therefore, you'll need to grant Tribuo those permissions
using a similar snippet when running with a security manager.

## Threat Model
As a library incorporated into other programs, Tribuo expects it's inputs to be
checked by the wider program; however, there are threats which are specific to
ML systems that can result in model or data leakage.

| Threat | Description | Exposed Assets | Possible Mitigations |
| ------ | ----------- | -------------- | -------------------- |
| Model replication | If an attacker can repeatedly query the model, where they either know or control the features, and they can observe the full prediction (e.g., the complete predicted probability distribution) for each query, then this can provide sufficient information for them to replicate the model.  If the model is considered an important asset, allowing an attacker to copy it could be detrimental. | The model parameters | Only return a small number of predictions (i.e., the top n) or do not provide the probability distribution. This slows down the attack, but does not completely prevent it. Other mitigations such as employing rate limiting or preventing the attacker from controlling or observing the feature inputs are necessary to fully prevent this attack.|
| Training metadata leak | The model file contains information about the training data such as the feature names, number of features, and number of examples. This information is potentially sensitive, as in the case of bigrams or trigrams from text. | Training metadata | Firstly, treat model files as confidential if the data itself is confidential. Secondly, use Tribuo's methods for one-way hashing of the feature names. Hashing prevents attackers from trivially discovering the features without needing to complete the process of supplying the input text and testing if the model output changes. Thirdly, other information present in the model file, such as the number of examples, can be redacted by removing the provenance information before the model is deployed. | 
| Training data leak | If an attacker can repeatedly query the model, it's possible for an attacker to find specific training data points that are part of the training data set. This attack is accomplished by measuring the confidence of the prediction (as training data points usually have a predicted confidence close to 1.0). | Training data | The most important mitigation is to treat model files as confidential if the training data is confidential. Once access to the model has been prevented, the mitigations for model replication apply. This attack is a variant of model replication that usually requires some foreknowledge of the identity of the training corpus. |



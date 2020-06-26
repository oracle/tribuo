# Security Considerations
Tribuo is a library which is designed to be incorporated into larger programs. We therefore
consider the trust boundary to be outside Tribuo, and to live somewhere in the larger program.
As such while we provide data loaders we expect them to be used on trusted or cleaned data, and
the larger program to control access to Tribuo's interfaces. Tribuo's data loaders and other
interfaces perform defensive copying and other standard procedures where user code crosses
over into Tribuo's internals, though this is not generally the case for performance reasons
in calls within Tribuo (e.g. the linear algebra library exposes mutable state to reduce copying).

## Serialized files
Currently Tribuo models are stored as Java serialized objects, and due to the inherent 
issues with Java serialization those files should only be loaded and saved to trusted 
locations where third parties do not have access. We have provided a [JEP 290](https://openjdk.java.net/jeps/290) 
[allowlist](jep-290-allowlist.txt) which will allow the deserialization of only classes found in the Tribuo library, and 
this should be enabled on the code paths which deserialize models or datasets. As
Tribuo supports Java 8+, and JEP 290 is an addition to the Java 8 API from 8u121, the best way to use 
the allowlist for those demos is by setting it as a process
wide flag. Additionally when running with a security manager Tribuo will need access to the relevant
filesystem locations to load or save model files, see the section on [Configuration](#Configuration)
for more details.

## Database access
Tribuo provides a SQL interface which can load data via a JDBC connection. As it's frequently
necessary to load data via a joined query, and from an unknown schema, Tribuo *does not* validate
the input SQL, it is expected that the program developer will do this as they know the schema they
are loading from. Tribuo supports connections via public key wallets via JDBC. To use this functionality
supply the wallet configuration to the JVM as a system property and use the constructors that accept
a java.util.Properties instance with the appropriate configuration.

## Native code
Tribuo uses several native libraries via JNI interfaces. Native code has different considerations 
to pure Java code, as it can cause issues in the running JVM. We are active contributors to all the
native ML libraries that Tribuo uses and fix issues upstream if we find them. Nevertheless you should
think carefully before running a model that requires native code inside an application container like a
JavaEE or JakartaEE server. Multiple instances of Tribuo running inside separate containers may cause
issues with JNI library loading, due to ClassLoader security considerations.

## Configuration
Tribuo uses [OLCUT](https://github.com/oracle/olcut)'s configuration and provenance systems which use reflection
to construct and inspect classes.  Therefore when running with a Java security
manager you need to give the OLCUT jar appropriate permissions. We have tested
this set of permissions which allows the configuration and provenance systems
to work:

    // OLCUT permissions
    grant codeBase "file:/path/to/olcut/olcut-core-5.1.3.jar" {
            permission java.lang.RuntimePermission "accessDeclaredMembers";
            permission java.lang.reflect.ReflectPermission "suppressAccessChecks";
            permission java.util.logging.LoggingPermission "control";
            permission java.io.FilePermission "<<ALL FILES>>", "read";
            permission java.util.PropertyPermission "*", "read,write";
    };

The read FilePermission can be restricted to the jars which contain configuration 
files, configuration files on disk, and the locations of serialised objects. The 
one here provides access to the complete filesystem, as the necessary read 
locations are program specific and should thus be narrowed based on your 
requirements. If you need to save an OLCUT configuration then you will also 
need to add write permissions for the save location.

Similar read and write permissions are necessary for Tribuo to be able to load and
save models, so a similar snippet will be needed for the Tribuo jar when running with
a security manager.

## Threat Model
As a library incorporated into other programs, Tribuo expects it's inputs to be 
checked by the wider program, however there are threats which are specific to ML systems which
can result in model or data leakage.

| Threat | Description | Exposed Assets | Possible Mitigations |
| ------ | ----------- | -------------- | -------------------- |
| Model replication | Repeated queries with known features, where the attacker receives the full predicted probability distribution for each query can provide enough information for the attacker to replicate the model. If the model is considered an important asset then allowing an attacker to copy it could be detrimental. | The model parameters | To mitigate this only return a small number of predictions (i.e. the top n) or do not provide the probability distribution. This slows down the attack, though doesn't completely prevent it. Other mitigations such rate limiting or preventing the attacker from controlling or observing the feature inputs are necessary to fully prevent it.|
| Training metadata leak | If an attacker has the model file, this contains information about the training data (such as feature names, number of features, number of examples etc) which could be potentially sensitive (e.g. bigrams or trigrams from text). | Training metadata | First, treat model files as confidential if the data itself was confidential. Second, Tribuo provides a method for one-way hashing all the feature names, which prevents attackers from trivially finding out the features without supplying input text and testing if the model output changes. Third, the other information such as number of examples can be redacted by removing the provenance information before the model is deployed. | 
| Training data leak | If an attacker can repeatedly query the model it's possible for an attacker to find specific training data points that are part of the training data set by measuring the confidence of the prediction (as training data points usually have a predicted confidence close to 1.0). | Training data | First, treat model files as confidential if the training data is confidential. Then the mitigations for model replication apply. This attack is a variant of model replication but usually requires a little more idea of what the training corpus is than a model replication attack. |

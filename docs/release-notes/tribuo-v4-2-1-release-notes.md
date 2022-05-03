# Tribuo v4.2.1 Release Notes

Small patch release for three issues:

- Ensure K-Means thread pools shut down when training completes ([#224](https://github.com/oracle/tribuo/pull/224))
- Fix issues where ONNX export of ensembles, K-Means initialization and several tests relied upon HashSet iteration order ([#220](https://github.com/oracle/tribuo/pull/220),[#225](https://github.com/oracle/tribuo/pull/225))
- Upgrade to TF-Java 0.4.1 which includes an upgrade to TF 2.7.1 which brings in several fixes for native crashes operating on malformed or malicious models ([#228](https://github.com/oracle/tribuo/pull/227))

OLCUT is updated to 5.2.1 to pull in updated versions of jackson & protobuf ([#234](https://github.com/oracle/tribuo/pull/234)). Also includes some docs and a small update for K-Means' `toString` ([#209](https://github.com/oracle/tribuo/pull/209), [#211](https://github.com/oracle/tribuo/pull/211), [#212](https://github.com/oracle/tribuo/pull/212)).

## Contributors

- Adam Pocock ([@Craigacp](https://github.com/Craigacp))
- Geoff Stewart ([@geoffreydstewart](https://github.com/geoffreydstewart))
- Yaliang Wu ([@ylwu-amzn](https://github.com/ylwu-amzn))
- Kaiyao Ke ([@kaiyaok2](https://github.com/kaiyaok2))


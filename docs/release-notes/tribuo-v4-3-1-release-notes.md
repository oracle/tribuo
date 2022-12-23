# Tribuo v4.3.1 Release Notes

Small patch release to bump some dependencies and pull in minor fixes. The most
notable fix allows CART trees to generate pure nodes, which previously they had
been prevented from doing. This will likely improve the classification tree
performance both for single trees and when used in an ensemble like
RandomForests.

- FeatureHasher should have an option to not hash the values, and TokenPipeline should default to not hashing the values ([#309](https://github.com/oracle/tribuo/pull/309)).
- Improving the documentation for loading multi-label data with CSVLoader ([#306](https://github.com/oracle/tribuo/pull/306)).
- Allows Example.densify to add arbitrary features ([#304](https://github.com/oracle/tribuo/pull/304)). 
- Adds accessors to ClassifierChainModel and IndependentMultiLabelModel so the individual models can be accessed ([#302](https://github.com/oracle/tribuo/pull/302)).
- Allows CART trees to create pure leaves ([#303](https://github.com/oracle/tribuo/pull/303)).
- Bumping jackson-core to 2.14.1, jackson-databind to 2.14.1, OpenCSV to 5.7.1 (pulling in the fixed commons-text 1.10.0).

## Contributors

- Adam Pocock ([@Craigacp](https://github.com/Craigacp))
- Jeffrey Alexander ([@jhalexand](https://github.com/jhalexand))
- Jack Sullivan ([@JackSullivan](https://github.com/JackSullivan))
- Philip Ogren ([@pogren](https://github.com/pogren))

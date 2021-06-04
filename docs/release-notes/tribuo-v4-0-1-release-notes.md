# Tribuo v4.0.1 Release Notes

This release fixes a few issues we found in the tutorials just before launch.

The IDXDataSource was added as an alternative way to load MNIST as the LibSVM
website (which the tutorial was originally based on) was intermittently down
during our pre-launch period.

- Fixes an issue where the CSVReader wouldn't read files with extraneous newlines at the end.
- Adds an IDXDataSource so we can read [IDX](http://yann.lecun.com/exdb/mnist/) (i.e. MNIST) formatted datasets.
- Updated the configuration tutorial to read MNIST from IDX files rather than libsvm files.

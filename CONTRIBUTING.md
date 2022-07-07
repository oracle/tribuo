# Contributing

We welcome your contributions! There are multiple ways to contribute.

## Issues
For bugs or enhancement requests, please file a GitHub issue unless it's security related. When filing a bug remember that the better written the bug is, the more likely it is to be fixed. If you think you've found a security vulnerability, do not raise a GitHub issue and follow the instructions on our [Security Policy](./SECURITY.md). 

## Contributing Code

We welcome your code contributions. To get started, you will need to sign the [Oracle Contributor Agreement](https://oca.opensource.oracle.com/) (OCA).

For pull requests to be accepted, the bottom of your commit message must have
the following line using the name and e-mail address you used for the OCA.

```
Signed-off-by: Your Name <you@example.org>
```

This can be automatically added to pull requests by committing with:

```
git commit --signoff
```

Only pull requests from committers that can be verified as having
signed the OCA can be accepted.

### Pull request process

1. Fork this repository
1. Create a branch in your fork to implement the changes. We recommend using
the issue number as part of your branch name, e.g., `1234-fixes`
1. Ensure that any Javadoc and/or documentation is updated with the changes.
1. Add a test for the new behaviour (or that exercises the bug if a bug fix).
1. Submit the pull request. *Do not leave the pull request text blank*. Explain exactly
what your changes are meant to do and provide simple steps on how to validate
your changes. Ensure that you reference the issue you created as well. The PR 
name will be the name of the squashed commit to main.
1. We will assign the pull request to be reviewed before it is merged.

## Code of Conduct
Follow the [Golden Rule](https://en.wikipedia.org/wiki/Golden_Rule). More specific guidelines are in the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md)

# Developer Guide

```scikit-activeml``` is a library that executes the most important query strategies. It is built upon the well-known machine learning framework scikit-learn, which makes it user-friendly. The motivation is ...

## Introduction

### Intended Audience

This guide is intended primarily for developers who want to work on the ...

To get started, you’ll want to set up your Development Environment and make sure you understand the branching strategy described in the Version Control section and how to make a pull request. Testing is expected. Opinions about Coding Style are welcome!

### Getting Help

> TODO-MMUEJDE: Was wenn ich fragen habe? Immer alles über Github oder

If you have any questions at all, please reach out to other developers via the channels listed in

- E-Mail
- Slack channel
- Discord channel
- Github issues
- Reddit
- ...

### Roadmap

Our Roadmap is ...

### Issue Tracker

We use GitHub Issues as our issue tracker...

## Get Started

Before you can contribute to this project, you need to do some work.

### Setup development environment

There are several ways to create a local Python environment, such as virtualenv[], pipenv[], miniconda[], etc. One possible workflow is to install miniconda and use it to create a Python environment. And use pip to install packages in this environment.

#### Example with miniconda

Create a new python environment named scikit-activeml:

```bash
conda create -n scikit-activeml
```

To be sure that the correct env is active:

```bash
conda activate scikit-activeml
```

Then install the pip:

```bash
conda install pip
```

### Install Dependencies

> Dependencies
> - joblib>=1.0.1
> - numpy>=1.20.1
> - scipy>=1.6.1
> - scikit-learn>=0.24.1
> - matplotlib>=3.3.4
> - iteration-utilities>=0.11.0

Now we can install some required dependencies for scikit-activeml, which are defined in the requirements.txt file.

```bash
# Make sure your scikit-activeml python env is active!
cd <project-root>
pip install -r requirements.txt
```

After the pip installation was successful, we can also install pandoc if it is not already installed.

```bash
# Check pandoc installation
```

Example: Macintosh (Homebrew)

```bash
brew install pandoc
```

Example: Linux (Homebrew)

Example: Windows (Homebrew)

## Contributing

### Project Structure

```
scikit-activeml
├── .github
│	└── main.yml
├── docs
│   ├── api
│   ├── example
│   ├── generated (Contains ..)
│   ├── logos
│   ├── notebooks
│   ├── conf.py
│   ├── refs.bib
│   └── generate.py
├── examples
│   ├── pool
│   └── stream
└── skactiveml
    ├── classifier
    │   └── tests
    ├── pool
    │   ├── tests
    │   └── multi
    │       └── tests
    ├── stream
    │   ├── tests
    │   └── budget_manager
    │       └── tests
    ├── tests
    ├── utils
    │   └── tests
    └── visualization
        └── tests
```

### Coding Conventions

As this library conforms to the convention of scikit-learn, the code should conform to PEP 8 Style Guide for Python Code. For linting, the use of flake8 is recommended.

- File organization
- Comments?
- Naming Conventions
  - Files
  - Classes
  - Functions
  - Vars
  - Maybe known python naming conventions?
- Writing Tests
- Commit Messages
  - https://chris.beams.io/posts/git-commit/
- etc.

### Testing

- Codecov
- How to run tests
- Test conventions
  - See Coding Conventions

### Lifecycle of a Pull Request

> TODO-MMUEJDE: Explain the common lifecycle...

### Local Builds

#### Build documentation (User Guide + Developer Guide)

```bash
# command
```

#### Build scikit-activeml

```bash
#  command
```

### Issue Tracking

We use GitHub[] ....

If you think you have found a bug in scikit-activeml, you can report it to the issue tracker. Documentation bugs can also be reported there. If you would like to file an issue about this devguide, please do so at the devguide repo.

#### Checking if a bug already exists

The first step before filing an issue report is to see whether the problem has already been reported. Checking if the problem is an existing issue will:

- help you see if the problem has already been resolved or has been fixed for the next release
- save time for you and the developers
- help you learn what needs to be done to fix it
determine if additional information, such as how to replicate the issue, is needed
- To do see if the issue already exists, search the bug database using the search box on the top of the issue tracker page.

#### Reporting an issue

- Use the following labels
  - documentation: If you ...
  - bug: if ...
  - cosmetics: if ...
  - feature: if ...
  - nice-to-have: if ...
  - ...
- Post error message
- python version
- dependency versions
- ...

If the problem you’re reporting is not already in the issue tracker, you need to log in by entering your user and password in the form on the left. If you don’t already have a tracker account, select the “Register” link or, if you use OpenID, one of the OpenID provider logos in the sidebar.
It is not possible to submit a bug report anonymously.
Once logged in, you can submit a bug by clicking on the “Create New” link in the sidebar.
The submission form has a number of fields, and they are described in detail in the Triaging an Issue page. 

This is a short summary:

- in the Title field, enter a very short description of the problem; less than ten words is good;
- in the Type field, select the type of your problem (usually behavior);
- if you know which Components and Versions are affected by the issue, you can select these too; otherwise, leave them blank;
- last but not least, you have to describe the problem in detail, including what you expected to happen, what did happen, and how to replicate the problem in the Comment field. Be sure to include whether any extension modules were involved, and what hardware and software platform you were using (including version information as appropriate).

#### Understanding the issue’s progress and status

The triaging team will take care of setting other fields, and possibly assign the issue to a specific developer. You will automatically receive an update each time an action is taken on the bug.

### Feature Development

> TODO-MMUEJDE: Wie läuft das ab?

- Add bibtex ref to...
- ...

### Example: Contribution cycle

> TODO-MMUEJDE: Brauchen wir sowas?

1. Fork the modAL repository and clone your fork to your local machine:

```bash
git clone git@github.com:username/modAL.git
```

2. Create a feature branch for the changes from the dev branch:

```bash
git checkout -b <feature, bug-fix, hotfix, etc...>/new-feature dev
```

> Make sure that you create your branch from dev.

3. After you have finished implementing the feature, make sure that all the tests pass. The tests can be run as

```bash
$ python3 path-to-modAL-repo/tests/core_tests.py
```

4. Commit and push the changes.

```bash
$ git add modified_files
$ git commit -m 'commit message explaning the changes briefly'
$ git push origin new-feature
```

## Deep Dive

### Handling of unlabeled instances

Active learning generally uses labeled and unlabeled instances. To simplify the data handling, the SkactivemlClassifier is able to handle unlabeled data. The unlabeled data is marked as such by setting corresponding entry in y (the label) during fitting to missing_label which is set during the initialization of the classifier. All classifier and the wrappers (e.g. for scikit-learn classifiers) are compatible with unlabeled instances.

### Handling of Batch / Non-Batch scenarios

All query strategies, except the stream based approaches, support the batch scenario. All strategies that are not explicitly designed to support the batch scenario shall employ a greedy strategy to iteratively select instances to fill the queried batch. The query methods have a batch_size parameter to specify the number of instances to queried instances. If the batch size is dynamic, the batch_size parameter shall be set to 'adaptive'. The utilities that are returned, when return_utilities is set to true, have the following shape: batch_size x n_cand, reflect the utilities for each individual acquisition.

### Handling of pool-based, stream-based AL and membership query synthesis

- separate packages and classes follow *PoolBasedQueryStrategy, *StreamBasedQueryStrategy, *MembershipQuerySynthesis

### Handling of active learning with multiple annotators

### Handling of uncertain oracles

### Transductive and inductive active learning

### Classification and regression

### Evaluation

### Stopping criteria

## Best Praticses

### Json example file structure

> TODO-MMUEJDE: Hier kann man paar habits mitgeben.

- Before contributing a new feature, please open a new issue. This helps us to discuss your idea and makes sure that you are not working in parallel with other contributors.

## Troubleshooting

> TODO-MMUEJDE: Bekannte Probleme auflisten und issues sparen

### Pandoc error on mac

```bash
error 0001: pandoc.py not found
```

A Solution is ...
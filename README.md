# ugbio-utils

This repository should include utils for bioinformatics pipelines.

The packaging management in this project is done using [Rye](https://rye.astral.sh)
### Rye

#### Install rye
1. `curl -sSf https://raw.githubusercontent.com/astral-sh/rye/main/scripts/install.sh | bash`
2. If on a personal laptop (not a remote instance accessed through ssh):<br>`echo '. "$HOME/.rye/env"' >> ~/.bashrc`
3. To activate rye in the current shell:<br>`. ~/.bashrc`
4. Configure rye to use uv (super fast):<br>`rye config --set-bool behavior.use-uv=true`

#### Install/Update the virtual environment
1. Change directory to `ugbio_utils` repo root.
2. `rye sync`

#### View project's state
1. Use `rye show` to get the current state of the project
2. Use `rye list` to get the currently installed packages


#### Run tests
1. Use `rye test --all` to run all workspaces' tests
2. Use `rye test --package [package_name]` to run tests of specific workspace

### Define a new ugbio workspace member
1. create a new <WORKSPACE_MEMBER_NAME> folder under `src` for the new workspace. this folder should contain:
    - Dockerfile
    - pyproject.toml 
    - README.<WORKSPACE_MEMBER_NAME>.md
    - ugbio_<WORKSPACE_MEMBER_NAME> folder for python src code
    - tests folder for python tests

2. Dockerfile can be based on ugbio_base image that contains common tools
3. In pyproject.toml you declare all the workspace's dependencies and requirments. It can contain also scripts to define executables. You must declare at least the follwoing `run_tests` script to allow run tests from the CI build:
    ```
    [project.scripts]
    run_tests = "pytest:main"
    ```
    
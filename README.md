# <Repository Title - Change to your own descriptive repository title >

[![](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)


  This is a template repo for python projects with pre-commit hooks and poetry and Dockerfile example # remove
  
  **Please change the pyproject.toml with the relevant information**

  **For the git pre-commit to work you must execute:**
  ```
  poetry run pre-commit install
  poetry run pre-commit install -t pre-commit
  ```
After the installation it will run the pre-commit hooks for all files changed as part of the commit.
This should look like this, notice mostly the red "Failed" issues that you must fix, the pre-commit verifies the fix before enables the commit:
```
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...........................................(no files to check)Skipped
check json...........................................(no files to check)Skipped
check for added large files..............................................Passed
ruff.................................................(no files to check)Skipped
ruff-format..........................................(no files to check)Skipped
mypy.................................................(no files to check)Skipped
poetry-check.............................................................Passed
poetry-check.............................................................Passed
[master 9a1a910e] Test pre-commit
 1 file changed, 1 deletion(-)
 ```
  
  <Add relevant information such as deployment information, links to design docs, release information, etc.>

## File System Visibility Overrides

**Directive:**
The agent is explicitly explicitly permitted to read and modify the following files, even if they appear in `.gitignore`:

* `Cargo.lock`

**Reasoning:**
These are local development configurations necessary for the agent to build the project, but they should not be committed to the remote repository.
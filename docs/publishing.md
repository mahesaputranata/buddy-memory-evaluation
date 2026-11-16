# Publish the prepared repository

Recommended new repository name: `buddy-memory-evaluation`. This is separate from the deployed Buddy code and easier to link from a CV. GitHub is the primary target; the same directory layout works on GitLab.

## GitHub CLI (Windows PowerShell)

Install Git and GitHub CLI if needed. Extract the project ZIP, open a terminal in `buddy-memory-evaluation`, and run:

```powershell
gh auth login
# Only if this extracted folder is not already a Git repository:
git init -b main
git add .
git diff --cached --stat
# Review staged files, especially benchmark utterances and frozen context packets.
git commit -m "Add offline memory retrieval and local LLM evaluation study"
gh repo create buddy-memory-evaluation --private --source . --remote origin --push
```

Private is a reviewable first publication. Once you have reviewed the data and selected a license, change visibility in repository settings if you want it public, then use its actual URL in the CV. No repository has been created by this package.

## Existing repository or GitLab

Create an empty repository on your chosen service, copy its actual remote URL, and add it as `origin` with `git remote add origin URL_FROM_YOUR_REPOSITORY`, then `git push -u origin main`. If origin already exists, inspect it with `git remote -v` rather than replacing it blindly. Do not paste the illustrative URL label literally.

Suggested GitHub description:

> Offline conversational-memory evaluation for a rover assistant: semantic retrieval, session filtering, optional reranking, and local Llama/Qwen answering experiments.

Suggested topics: `llm`, `rag`, `evaluation`, `conversational-memory`, `ollama`, `sentence-transformers`, `robotics`.

Recommended featured links: README, `docs/experiment_history.md`, and the cleaned notebook. The prepared copy retains research limitations and negative results; do not rename evidence coverage as answer accuracy.

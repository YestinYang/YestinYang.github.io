---
layout: post
title: Git and GitHub
key: 20180325
tags:
  - Udacity
  - Notes
  - Study
  - Linux
lang: en
mathjax: true
mathjax_autoNumber: true
---

### Git: Commands

- Create Repo
  - `git init` initial a folder as a new repo, tracking all modification under it
    - `.git/config` configure only for this repo
  - `git clone <path> [<new dir name>]`
    - cannot create nested repo, so do check your `pwd`
  - `git status`
- Review Commits
  - `git log` for commit history
    - `git log --oneline` show only 7 characters of SHA and commit
    - `git log --stat` show files changed for each commit
    - `git log --patch` or `git log -p` show the detailed changes of each commit
    - `git log -w` ignore the change of blank when showing patch info
    - `git log <7 character of SHA>` start from particular commit
  - `git show` show commit with knowing SHA
    - share all arguments of `git log`
- Make Commits
  - `git add` 
    - `git add .` to stage all folders and files
    - `git rm --cached <file>` to unstage
  - `git commit` open default editor
    - `git commit -m "commit message"` for short commit with only message
    - Rules of writing commits
      - Message: explain the role of submission
        - Good: "Update the footer to copyright information"
        - no why / no how / no and
      - Description: skip a line after message then start
  - `git diff` show changes of unstaged files
  - `.gitignore` file under repo to ignore specified files
- Develop on Branches
  - `git tag` show tag
    - `git tag -a v1.0 [<SHA>]` create a tag "v1.0" with comments (creator / date / message) for (optional) particular commit
    - `git tag -d "v1.0"` remove tag "v1.0"
    - use `git log` to verify the location of tag (use `--decorate` in older version)
  - `git branch` show all branches
    - `git branch <branch_name> [<SHA>]` create a new branch at (optional) particular commit
    - `git branch -d <branch_name>` delete branch (inactivate) w/o commit
      - to delete branch with commits, use `-D` instead
  - `git checkout` switch `HEAD` pointer to a branch, making it as active branch
    - convert file version to latest version of active branch (it replace all files under commit in previous branch, with that in the new branch)
    - `git checkout -b <new_branch_name> <create_on_which_branch>`  create a new branch and checkout it
    - ![Git Branches](https://github.com/YestinYang/Udacity_Data_Analyst/tree/master/img/git_branch.png){:.border}
  - `git log --oneline --graph --all` shows all branches and commits, also relationships
  - `git merge`
    - `git merge <other_branch>` merge a branch into current branch, like `master`
      - Fast-forward: moving `HEAD` to the latest commit (merge `social-links` to `master)`
      - Auto-merge: create a commit and merge (merge `sidebar` to `master`)
      - Merge conflicts: same row has been differently modified in both branches
        - git stop merging and return a single file, combining conflicting portions from both files, showing conflicts with indicators
        - update the file and commit it, then merge is completed automatically
        - you can just commit it and let merge done, but the conflicts will show in your file
      - ![Git Merge](https://github.com/YestinYang/Udacity_Data_Analyst/tree/master/img/git_merge.png){:.border}
    - `git reset --hard HEAD^` cancel merge
- Undo Changes
  - `git commit --amend` alter the most -recent commit
    - `git add` and run `git commit --amend` to add new files into last commit
  - `git revert <SHA>` create a new commit reversing given commit
  - Commit Reference
    - ![Quoting](https://github.com/YestinYang/Udacity_Data_Analyst/tree/master/img/git_father.png){:.border}
  - `git reset` erases commit
    - `git reset --mixed HEAD^^` move `HEAD` back to `HEAD^^`, and move changes of `HEAD^` into working dir (default)
    - `git reset --soft HEAD^^` move changes back to stage
    - `git reset --hard HEAD^^` delete changes
    - Backup strategy
      - before any `reset` operation, `git branch backup` 
      - after reset, if want to back to the starting point
        - `git checkout -- index.html` remove uncommited change to file "index.html"
        - `git merge backup` 

### Git: Shell Setting

1. [Lib of Setting optimized for Git](https://dotfiles.github.io/)
2. [Udacity style](https://classroom.udacity.com/nanodegrees/nd002-cn-advanced-career/parts/0f916a11-e0ae-4845-bb35-ee05bc7f1ac9/modules/63a3bbc1-bc00-4ee7-a894-a2d91841f30f/lessons/1b369991-f1ca-4d6a-ba8f-e8318d76322f/concepts/63a6f935-dea7-43c2-aaa3-61deea5070c8)

### GitHub: Remote Repo

- `git remote` show shortname of remote repo
  - `git remote -v` show detailed info
  - `git remote add <shortname> <repo_link>` add in repo link and its shortname
- `git push <remote_shortname> <remote_branch>` push all your local git info to remote repo
  - after `push` , `git log` will show the commit location of remote-tracking branch
- `git fetch <remote_shortname> <remote_branch>` download remote info and move remote-tracking 
- `git pull <remote_shortname> <remote_branch>` fetch and merge with local branch
- ![fetch and merge](https://github.com/YestinYang/Udacity_Data_Analyst/tree/master/img/git_fetch.png){:.border}

### GitHub: Collaborating with Fork

- `git shortlog` show commits under different collaborator
  - `git shortlog -s -n` show the number of commits only, sorted by large to small
- `git log --author="Paul Lewis"` show commits made by Paul Lewis
- `git log --grep="bug"` 
- How to contribute?
  - `Fork` the repo
  - Check CONTRIBUTION.md
  - To contribute large effort, use `Issues` to communicate with project maintenance before starting
  - Create branch with clear name meeting requirement
  - Create clear, frequent and small commits
  - Pull request

### GitHub: Keep Updated to Remote Repo

- Pull Request

  - ![pull request](https://github.com/YestinYang/Udacity_Data_Analyst/tree/master/img/git_pull_request.png){:.border}

- Fetch and Pull from Upstream

  - ![upstream](https://github.com/YestinYang/Udacity_Data_Analyst/tree/master/img/git_upstream.png){:.border}

- `git rebase` transfer commits onto a new base

  - `git rebase -i HEAD~3` squash last three commits interactively (refer to [YouTube](https://www.youtube.com/watch?time_continue=222&v=cL6ehKtJLUM) for detailed setting)
    - need force `git push` due to deleting commits
  - Backup strategy: create backup branch before `rebase` 
  - ![Rebase](https://github.com/YestinYang/Udacity_Data_Analyst/tree/master/img/git_rebase.png){:.border}
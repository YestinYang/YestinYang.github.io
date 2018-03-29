---
layout: post
title: Linux Command Line Basic and Shell
key: 20180313
tags:
  - Udacity
  - Udemy
  - Notes
  - Study
  - Linux
lang: en
mathjax: true
mathjax_autoNumber: true
---

## Go Into the Shell

**Environment:** VirtualBox + Vagrant + Git Bash

**Terminal and Shell:** 
- Terminal (emulator) displays your keyboard input and the output, but itself do not know how to handle your input
- Shell will accept the input transferred from Terminal, run the command and then send the output to Terminal to display
	- Default shell on Linux and Mac is GNU Bash
	- Can also use such as Python interpreter instead of shell

**Some Commands:** a bit like calling a Python function (but shell is used to run a program, function is used to organize a program)

```bash
date
expr 2 + 2		# run program, not organize program
echo You rock
uname	# print OS name
hostname
host udacity.com
bash --version		# most command has --version or -V
history		# typed commands
```

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-03-13_1520847868518.png){:.border}

## Shell Command

**`echo`:**
- `echo $COLUMNS x $LINES` outputs info with variable

**Command History:**
- Up arrow
- `history`
- Ctrl+R for used command search

**File Related:**
- `ls`, by default hide file starting with . (normally store cache or configuration which is not interesting)
	- `ls ~` for home dir
- `curl` for download anything
	- `curl -o dictionary.txt -L 'http://t.cn/RYkeaZi'`, always keep link inside single quotes
- `unzip`
- `cat` for concatenate, outputing multiple (small) files
- `less` for display file screen by screen
	- `/` for search
	- `q` for quit
- `grep` for search in file
- `nano` for edit the file (integrated in ubuntu)
- `wc` for word counts
- `diff` for version comparison
- `rm`
	- `-i` for interactive to confirm
	- `-r` for recursively delete all files under dir
	- `-f` for ignoring warning 
- `apropos` for list of commands related to particular keywords, for example, `apropos working directory`

**`man + command` command:** get detailed information of certain command

**PS:** argument within [] is optional; and arguments are case sensitive

**Line Based Programs:** like `ping` (stop with Ctrl+C), `sort` (execute with Ctrl+D), `bc` (quit or Ctrl+D)

**Full Screen Interactive Programs:** like `man`, `less` for displaying long file

**Output Result to a File:**

```bash
echo 'Hello World' >> demo.txt		# append to demo.txt
echo 'Brand New' > demo.txt		# overwrite demo.txt
```

**Pipe Commands:**

```bash
grep ibo dictionary.txt | less		# search ibo in a txt file, and packages the output and send to less
curl -L 'https://tinyurl.com' | grep fish		# download from a url and search fish in this file
```

**Variable:**
- Define just like that in Python, and call it with `$` (variable interpolation)
- Shell variable: like `$COLUMNS`
- Environment variable: like `$PWD` or `$LOGNAME`
	- `$PATH` variable stores path of your program files, like path of `ls`
		- For the exact path of `ls`, use `which ls`

## The Linux Filesystem

**File Name:** do not have requirement of file name, and even not require a file extension like Windows

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-03-13_1520862146223.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-03-13_1520862271210.png){:.border}

**Working Directory:** Linux only has one drive:root, to store all files
- `pwd` for print working directory
- `cd` for change directory
	- `cd ..` for go outer dir
	- `cd` only for go to your home dir
- `mkdir` for create dir
- `rmdir` for remove EMPTY dir
	- `rm -r` to remove entities in dir

**Absolute and Relative Path:** start with `/` or not
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-03-13_1520862884202.png){:.border}

**Tab Completion:** for file name and also dir
- One hit for completion, if there are more than one return, press Tab again to show all possible returns

**Moving and Copying Files:** `mv` and `cp`
- `touch` for create file

**Globbing:** 通配符
- `*` for anything, can be nothing also
- `?` for any one character
- `{item_1, item_2, ..}` for multiple
- `[aeiou]` for any one character

## Simple Bash Script

- Add the path of Notepad++ into SYSTEM PATH of windows
- Record down the path of bash by `which bash` (below takes /bin/bash as example)
- Run `notepad++ file.sh`
- In Notepad++

```bash
#!/bin/bash

echo 'Hello World'
```

- Check flag by `ls -l`, `x` in `-rwxr-xr-x` is executable
	- If not, use `chmod +x file.sh`
- Run with `./file.sh`

## Customize Your Shell

**.bash_profile:** run every time when shell start (or run `source .bash_profile`)
- can add in PATH of your own shell script, so that run it directly
- can display certain words when start

**`$PS1` Variable:** for appearance of shell
- [PS1 style generator](http://bashrcgenerator.com/)
- setting with `PS1 = "<code generated>"`

**`alias`:** customize your command (can put into .bash_profile to fix)

```bash
alias ll = 'ls -la'
alias cl = 'curl -L'
```
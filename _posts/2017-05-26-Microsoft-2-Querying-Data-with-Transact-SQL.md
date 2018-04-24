---
layout: post
title: Querying Data with Transact-SQL - Microsoft Certificate in Data Science 2 
key: 20170526
tags:
  - Microsoft
  - Notes
  - Study
  - SQL
  - Data Science
lang: en
---







## 1. Introduction to Transact-SQL

- What is Transact-SQL

  - Structured Query Language (SQL)
  - T-SQL is for SQL server and Azure SQL Database
  - SQL is declarative, not procedural -- describe what you want, don't specify steps

- Relational Databases *(like the relationship used in Power Pivot between different tables)*

  - Entities are represented as relations (tables), in which their attributes are represented as domains (columns)


  - Most relational databases are normalized, with relationships defined between tables through primary and foreign keys
    - ![@Structure of This Course](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-05-26_E52EF56E1041.png){:.border}

- Schemas and Object Names

  - Schemas are namespaces for database objects (tables)

    - Fully-qualified names ([ ] means optional)

        - ```sql
          [server_name.][database_name.][schema_name.]object_name
          ```

        - ![@Structure of This Course](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-05-26_689F8A592354.png){:.border}

        - in most cases we abbreviate into `schema_name.object_name` 

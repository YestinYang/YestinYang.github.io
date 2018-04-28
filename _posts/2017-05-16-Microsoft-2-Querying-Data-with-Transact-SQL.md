---
layout: post
title: Querying Data with Transact-SQL - Microsoft Certificate in Data Science 2 
key: 20170516
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
  - ![](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-05-16_E52EF56E1041.png){:.border}

- Schemas and Object Names
  - Schemas are namespaces for database objects (tables)

  - Fully-qualified names ([ ] means optional)

      - ```sql
        [server_name.][database_name.][schema_name.]object_name
        ```

      - ![](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-05-16_689F8A592354.png){:.border}

      - in most cases we abbreviate into `schema_name.object_name` 

- Types
    - Data Manipulation Language (DML) -- for querying and modifying data (mainly focus on)
      - Data Definition Language (DDL) -- for defining database objects
      - Data Control Language (DCL) -- for assigning security permissions


### The SELECT Statement
  - Sequence of Statement Running

    - ![](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-05-16_9605A65C32DC.png){:.border}
    - Main logic here is that grab data from certain source, and do conditioning, filtering and sorting

  - Basic Examples

    - ```sql
      -- All columns (not recommended)
      SELECT * FROM Production.Product;
      ```

    - ```sql
      -- Specific columns
      SELECT Name, ListPrice
      FROM Production.Product;
      ```

    - ```sql
      -- Expressions and Aliases
      SELECT Name AS Product, ListPrice * 0.9 AS SalePrice
      FROM Production.Product
      ```

### Working with Data Types

  - ![](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-05-16_459E8DC2A653.png){:.border}
  - Implicit Conversion: compatible data types can be automatically converted between subtypes
  - Explicit Conversion: requires an explicit conversion function
    - `CAST` 
    - `CONVERT` / `PARSE` has options to formatting data
    - `TRY_CAST` /  `TRY_CONVERT`  / `TRY_PARSE` return NULL instead of stop due to error
    - `STR` number to string



### Working with NULLs

- NULL represents a missing or unknown value

- ANSI behavior for NULL values

  - The result of any expression containing a NULL value is NULL
    - `2 + NULL = NULL`
    - `'MyString:' + NULL = NULL`
  - Equality comparisons always return false for NULL values
    - `NULL = NULL` returns false (unknown cannot be equal to unknown)
    - `NULL IS NULL` returns true (unknown is unknown)

- `ISNULL` column / variable, value

  - Returns value if the column or variable is NULL

  - ```sql
    -- If cannot CAST AS integer , return 0 instead of NULL
    -- NULL numbers = 0
    SELECT Name, ISNULL(TRY_CAST(Size AS integer),0) AS NumericSize
    FROM SalesLT.Product;
    ```

  - ```sql
    -- If Color/Size is NULL, return blank instead of NULL
    -- NULL strings = blank string
    SELECT ProductionNumber, ISNULL(Color,'') + ', ' + ISNULL(Size,'') AS NumericSize
    FROM SalesLT.Product;
    ```

  - `NULLIF` column/variable, value

    - ```sql
      -- If Color is Multi, make it as NULL is output
      -- Multi color = NULL
      SELECT Name, NULLIF(Color, 'Multi') AS SingleColor
      FROM SalesLT.Product;
      ```

  - `CASE` , like `if-else` in python

    - ```sql
      /* Select Name, and then go to the other column:
      if SellEndDate is NULL, then output On Sale (can use ISNULL function),
      otherwise Discontinued, in SalesStatus column */
      SELECT Name,
      	CASE
      		WHEN SellEndDate IS NULL THEN 'On Sale'
      		ELSE 'Discontinued'
      	END AS SalesStatus
      FROM SalesLT.Product;
      ```

    - ```sql
      /* If Size is number, output number;
      if Size is NULL, output n/a */
      SELECT Name,
      	CASE Size
      		WHEN 'S' THEN 'Small'
      		WHEN 'M' THEN 'Medium'
      		WHEN 'L' THEN 'Large'
      		WHEN 'XL' THEN 'Extra-Large'
      		ELSE ISNULL(Size, 'n/a')
      	END AS ProductSize
      FROM SalesLT.Product;
      ```

## 2. Querying Tables with SELECT

### Removing Duplicates

- `SELECT ALL` -- return all rows

- `SELECT DISTINCT` -- return unique only

- ```sql
  SELECT DISTINCT Color
  FROM Production.Product;
  ```

### Sorting Resulting

- ```sql
  /* ORDER BY
  Can use unselected columns for sorting */
  SELECT ProductCategory AS Category, ProductName
  FROM Production.Product
  ORDER BY Category, Price DESC;
  ```

- ```sql
  /* TOP -- limiting sorted results
  SELECT TOP (N) | TOP (N) Percent
  SELECT TOP (N) WITH TIES */
  SELECT TOP 100 Name, ListPrice
  FROM SalesLT.Product
  ORDER BY ListPrice DESC;
  ```

- ```sql
  /* OFFSET and FETCH -- paging through result like bbs pages
  (dependent on ORDER BY clause) */
  SELECT Name, ListPrice
  FROM SalesLT.Product
  ORDER BY ProductNumber
  OFFSET 10 ROWS
  FETCH FIRST 10 ROWS ONLY;
  ```


### Filtering and Predicates

- ![](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2017-05-16_4723FB1FFD26.png){:.border}

- `LIKE` with glob

  - ```sql
    -- List info about products taht have a product number beginning FR
    SELECT productnumber, Name, ListPrice
    FROM SalesLT.Product
    WHERE ProductNumber LIKE 'FR%';
    -- % for no matter length
    ```

  - ```sql
    -- Filter the previous query to ensure that the product number contains two sets of two digits
    SELECT Name, ListPrice
    FROM SalesLT.Product
    WHERE ProductNumber LIKE 'FR-_[0-9][0-9]_-[0-9][0-9]';
    -- _ for any one character
    ```

- Combining Multiple Conditions

  - ```sql
    -- Find products that have a category ID of 5,6,7 and have a sell end date
    SELECT ProductCategoryID, Name, ListPrice, SellEndDate
    FROM SalesLT.Product
    WHERE ProductCategoryID IN (5,6,7) AND SellEndDate IS NULL;
    ```

## 3. Querying Multiple Tables with Joins (Columns)


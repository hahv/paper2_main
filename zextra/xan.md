https://github.com/medialab/xan
# Most Common `xan` Commands – Quick Reference

| Category              | Command              | Example Command                                      | Description                                      |
|-----------------------|----------------------|------------------------------------------------------|--------------------------------------------------|
| **Preview**           | `view`               | `xan view file.csv`                                  | Pretty table preview in terminal                 |
| **Preview**           | `view` (semicolon)   | `xan view -d ';' file.csv`                           | Preview CSV with `;` separator                   |
| **Preview**           | `flatten`            | `xan flatten file.csv`                               | Show one row as key-value list                   |
| **Info**              | `headers`            | `xan headers file.csv`                               | List all column names                            |
| **Info**              | `count`              | `xan count file.csv`                                 | Count number of rows                             |
| **Filter**            | `filter`             | `xan filter 'age > 30' file.csv`                     | Filter rows using expression                     |
| **Filter**            | `search`             | `xan search -s name "John" file.csv`                 | Search text in a specific column                 |
| **Slice**             | `head`               | `xan head -n 50 file.csv`                            | Show first N rows                                |
| **Slice**             | `tail`               | `xan tail -n 20 file.csv`                            | Show last N rows                                 |
| **Select Columns**    | `select`             | `xan select id,name,age file.csv`                    | Keep only selected columns                       |
| **Sort**              | `sort`               | `xan sort -s age file.csv`                           | Sort by column(s)                                |
| **Frequency**         | `frequency`          | `xan frequency -s gender file.csv`                   | Count unique values + percentages                |
| **Statistics**        | `stats`              | `xan stats -s salary,age file.csv`                   | Basic statistics (mean, min, max, stddev, etc.)  |
| **Group & Aggregate** | `groupby`            | `xan groupby city 'mean(salary)' file.csv`           | Group by column and aggregate                    |
| **Create Column**     | `map`                | `xan map 'age * 2 as double_age' file.csv`           | Create new column from expression                |
| **Modify Column**     | `transform`          | `xan transform name 'upper(name)' file.csv`          | Modify existing column                           |
| **Deduplicate**       | `dedup`              | `xan dedup -s email file.csv`                        | Remove duplicate rows                            |
| **Join**              | `join`               | `xan join id users.csv id orders.csv`                | Join two CSV files                               |
| **Change Delimiter**  | `fmt`                | `xan fmt -d ';' file.csv`                            | Change output delimiter (e.g. to semicolon)      |

### Most Useful One-Liners

```bash
# 1. Quick preview (especially for semicolon CSV)
xan view -d ';' data.csv

# 2. See column names
xan headers data.csv

# 3. Basic statistics
xan stats -s price,quantity data.csv

# 4. Frequency table
xan frequency -s category data.csv | xan view

# 5. Filter and preview
xan filter 'price > 1000' data.csv | xan view

# 6. Sort and show
xan sort -s price file.csv | xan view
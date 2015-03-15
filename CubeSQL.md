SQL-based language for querying cubes.

Go to [CubeSQL.ebnf](CubeSQL.ebnf) for formal definition.

## Example queries

```sql
SELECT
	locality, yob, SUM(signatures)
FROM
	elections2015
WHERE
	yob IN <1980, 1997>
AND
	locality IN {"Warszawa", "Kraków"}
AND
	pesel[10] IN {1, 3, 5, 7, 9}
ORDER BY
	SUM(signatures) DESC
LIMIT
	5
```

Note: GROUP BY is probably unnecessary, as we can automatically group by all selected dimensions (and merge everything if no dimension is selected).

```sql
CREATE CUBE elections2015
{
	dim recv TIME,

	dim yob <1900,2015>,
	dim commune CHAR[7],
	dim locality TEXT,
	dim pesel CHAR[11],

	mea signatures <0,1000000>,
	mea applications <0,100000>,
};
```

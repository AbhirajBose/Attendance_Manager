# DECISIONS.md  (about 150-200 words in total - plain language, your own words)

1. Indexes - I made a unique(emp_code,date) index on the attendance logs, which stops duplicate days and speeds up monthly and department lookups. Additionally, I made (date,emp_code) for the list order and month scans, (status,date,emp_code) for status filters, and (emp,code,punch_in) to find the open record at punch-out. I made a unique emp_code index,plus(department,emp_code) and (joined_on,department) on employees for headcount. I rejected an index on late_minutes: one month is a small range, so it was not worth the slower writes.

2. Punch-in race - Both requests build the same record and call insert. The database's unique index lets only one succeed. The other fails with a duplicate-key error, and my code turns that into a 409.

3. Ties - The leaderboard uses a rank window, so tied people share a rank and the next rank is skipped. I apply the limit to the rank, so everyone tied at the cutoff is returned, even if that is more rows than the limit.
4. Headcount -  The summary starts from employees who joined by month end, then attaches their logs. A person with no logs still counts as one.
5. 100x data - I would store daily totals per department, updated on each punch, so analytics stop scanning raw records.

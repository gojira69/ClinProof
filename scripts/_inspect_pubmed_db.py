import sqlite3

conn = sqlite3.connect('data/pubmed_index/pubmed_meta.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', tables)
for t in tables:
    tname = t[0]
    cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
    print(f'\nTable: {tname}')
    print('Columns:', [(c[1], c[2]) for c in cols])
    try:
        rows = conn.execute(f'SELECT * FROM "{tname}" LIMIT 3').fetchall()
        for r in rows:
            print('  row:', str(r)[:200])
    except Exception as e:
        print('  (cannot select):', e)
conn.close()

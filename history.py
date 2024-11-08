import os
import json
import sqlite3
import time

from pathlib import Path
from typing import List
from dataclasses import asdict, fields

from methods import MethodResult

METHOD_HISTORY = Path(__file__).parent / 'history.sqlite'

class HistoryDB:
    table_name = 'completed_methods'
    table_definition = f"""\
        CREATE TABLE IF NOT EXISTS {table_name}(
            id TEXT PRIMARY KEY,
            name TEXT
            source TEXT
            method_data JSON
            log JSON
            result JSON
            created TIMESTAMP
            finished TIMESTAMP
        );"""
    columns = [f.name for f in fields(MethodResult)]

    def __init__(self, database_path: str = METHOD_HISTORY) -> None:
        self.db_path = database_path
        self.db = None
        self.open()

    def open(self) -> None:
        db_exists = os.path.exists(self.db_path)
        self.db = sqlite3.connect(self.db_path)
        if not db_exists:
            self.db.execute(self.table_definition)

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def smart_insert(self, result: MethodResult) -> None:
        """Inserts or, if result already exists, updates a result. Uses task ID as unique identifier

        Args:
            result (MethodResult): result to update or insert into the db
        """

        res = self.db.execute(f"""\
            INSERT INTO {self.table_name}({','.join(self.columns)}) VALUES ({','.join(['?']*len(self.columns))})
            ON CONFLICT(id) DO UPDATE SET 
                {','.join([f'{c}=excluded.{c}' for c in self.columns if c != 'id'])};
        """, [getattr(result, c) for c in self.columns])

        self.db.commit()
    
    def search_id(self, id: str, case_sensitive: bool = False) -> List[MethodResult]:
        res = self.db.execute(f"SELECT * FROM {self.table_name} WHERE name {'LIKE' if case_sensitive else 'ILIKE'} ?", (f'%{id}%',))
        materials = res.fetchall()
        return [MethodResult(*m) for m in materials]

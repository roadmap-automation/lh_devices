import asyncio
import os
import json
import sqlite3

from pathlib import Path
from dataclasses import fields

from .methods import MethodResult

METHOD_HISTORY = Path(__file__).parent / 'history.sqlite'

def json_format(field_value, field_type):
    return json.dumps(field_value) if (field_type is dict) | (field_type is list) else field_value

def json_rehydrate(field_value, field_type):
    return json.loads(field_value) if (field_type is dict) | (field_type is list) else field_value

class HistoryDB:
    table_name = 'completed_methods'
    table_definition = f"""\
        CREATE TABLE IF NOT EXISTS {table_name}(
            id TEXT PRIMARY KEY,
            created_time TIMESTAMP,
            source TEXT,
            method_name TEXT,
            method_data JSON,
            finished_time TIMESTAMP,
            log JSON,
            result JSON
        );"""
    columns = {f.name: f.type for f in fields(MethodResult)}

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
            INSERT INTO {self.table_name}({','.join(self.columns.keys())}) VALUES ({','.join(['?']*len(self.columns.keys()))})
            ON CONFLICT(id) DO UPDATE SET 
                {','.join([f'{c}=excluded.{c}' for c in self.columns.keys() if c != 'id'])};
        """, [json_format(getattr(result, c), ctype) for c, ctype in self.columns.items()])

        self.db.commit()
    
    def search_id(self, id: str) -> MethodResult:
        res = self.db.execute(f"SELECT * FROM {self.table_name} WHERE id=?", (id,))
        record = res.fetchone()
        return MethodResult(*[json_rehydrate(f, ftype) for f, ftype in zip(record, self.columns.values())]) if record is not None else None

class DatabasePlugin:
    """Plugin for adding database functionality
    """

    def __init__(self, database_path: Path | str = None):

        self.database_path = database_path

    async def async_save_to_database(self, result: MethodResult):
        """Saves a method result to the database, only if a task id is associated with it

        Args:
            result (MethodResult): result to save
        """

        await asyncio.to_thread(self.save_to_database, result)

    def save_to_database(self, result: MethodResult):
        """Saves a method result to the database, only if a task id is associated with it

        Args:
            result (MethodResult): result to save
        """

        if result.id is not None:
            with HistoryDB(self.database_path) as db:
                db.smart_insert(result)

    def read_from_database(self, id: str) -> MethodResult | None:
        """Reads a method result from the database

        Args:
            id (str): id of record

        Returns:
            MethodResult | None: MethodResult object if id exists, otherwise None
        """

        with HistoryDB(self.database_path) as db:
            return db.search_id(id)
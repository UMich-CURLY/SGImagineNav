# test db: json
import time
import torch
import numpy as np
import datetime
import json
import pickle
import ibis
from ibis import _
import io

def generate_test_data():
    return {
        'step':np.random.randint(0, 501),
        'success':np.random.randint(0, 2),
        'np_feat': np.random.rand(500,3).astype(np.float32),
        'torch_feat': torch.rand(500,3),
        'info': {'score': np.random.rand()},
        'time': datetime.datetime.now(),
        'scene_graph_json': {'room': 'living room', 'objects': ['sofa', 'table']},
        'some_dict': {'room': 'living room', 'objects': ['sofa', 'table']},
        'some_complicated_dict': {
            'room': 'living room',
            'objects': ['sofa', 'table'],
            'feats': [{'clip':torch.rand(500, 3)}, {'clip':torch.rand(500, 3)}]
        },
        'some_list': ['aaa', 'bbb', 'ccc'],
        'some_complicated_list': [torch.rand(500, 3), np.random.rand(500, 3)],
        'good_byte': b'good bye',
    }

def set_interactive(interactive):
    ibis.options.interactive = interactive

def to_ibis_type(key, value):
    schema_list = []
    if isinstance(value, float):
        schema_list.append((key, "float64"))
    elif isinstance(value, int):
        schema_list.append((key, "int64"))
    elif key.endswith('_json'):
        schema_list.append((key, "json"))
    elif isinstance(value, str):
        schema_list.append((key, "string"))
    elif isinstance(value, bytes):
        schema_list.append((key, "bytes"))
    elif isinstance(value, datetime.datetime):
        schema_list.append((key, "timestamp"))
    elif isinstance(value, np.ndarray):
        schema_list.append((key+'_tensor', 'bytes'))
        # schema_list.append((key, f'array<{value.dtype.name}>'))
        # schema_list.append((key+'_shape', 'array<int>'))
    elif isinstance(value, torch.Tensor):
        schema_list.append((key+'_tensor', 'bytes'))
        # schema_list.append((key, f'array<{value.detach().cpu().numpy().dtype.name}>'))
        # schema_list.append((key+'_shape', 'array<int>'))
    elif isinstance(value, np.generic):
        schema_list.append((key, value.dtype.name))
    elif isinstance(value, dict) or isinstance(value, list):
        try:
            json.dumps(value)
            schema_list.append((key, "json"))
        except TypeError:
            schema_list.append((key+'_pickle', "bytes"))
    elif value is None:
        print(f"Example data is None: {key}={value}")
    else:
        raise ValueError(f"Unsupported data type: {key}={type(value)}")
    return schema_list

def to_ibis_value(key, value):
    new_data = {}
    if key.endswith('_json'):
        new_data[key] = value
    elif isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
        buf = io.BytesIO()
        np.save(buf, value)
        new_data[key+'_tensor'] = buf.getvalue()
        buf.close()
        # new_data[key] = value.flatten()
        # new_data[key+'_shape'] = list(value.shape)
    elif isinstance(value, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, value)
        new_data[key+'_tensor'] = buf.getvalue()
        buf.close()
        # new_data[key] = value.flatten()
        # new_data[key+'_shape'] = list(value.shape)
    elif isinstance(value, list) or isinstance(value, dict):
        try:
            new_data[key] = json.dumps(value)
        except TypeError:
            new_data[key+'_pickle'] = pickle.dumps(value)
    elif value is None:
        pass
    else:
        new_data[key] = value
    return new_data


def get_df(db_path, table_name, filter=None, select=None):
    con = connect_db(db_path, read_only=True, retry_interval=0.001)
    try:
        table = con.table(table_name)
        if filter is not None:
            table = table.filter(filter(_))
        if select is not None:
            table = table.select(select)
        df = table.execute()
        return df
    except Exception:
        raise
    finally:
        con.disconnect()

def get_data(db_path, table_name, filter=None, select=None):
    df = get_df(db_path, table_name, filter=filter, select=select)
    new_data_list = []
    for i in range(len(df)):
        new_data = {}
        for k,v in df.loc[i].items():
            if v is None:
                new_data[k] = v
            elif k.endswith('_tensor'):
                new_data[k[:-7]] = np.load(io.BytesIO(v))
            elif k.endswith('_shape'):
                pass
            elif k+'_shape' in df:
                new_data[k] = np.array(v, dtype=np.float32).reshape(df.loc[i][k+'_shape'])
            elif k.endswith('_pickle'):
                new_data[k[:-7]] = pickle.loads(v)
            else:
                new_data[k] = v
        new_data_list.append(new_data)
    return new_data_list

def generate_schema(data):
    """
    Generate a schema for the given data. Make sure to include all the keys in the data.
    @ note: add '_json' to the key if you want to store the data as json
    @ note: torch.Tensor and np.ndarray will be flattened and an extra column '{key}_shape' will be added
    @ note: list
    """
    schema_list = []
    if isinstance(data, list):
        data = data[0]
    for key, value in data.items():
        schema_list += to_ibis_type(key, value)
    return ibis.schema(schema_list)

def update_schema(con, table_name, new_schema):
    table_data = con.table(table_name)
    merged_schema = {k:v for k, v in con.table(table_name).schema().items()}
    schema_changed = False
    for k, v in new_schema.items():
        if k not in merged_schema:
            print(f"[Database] Adding new column: {k} with type {v}")
            merged_schema.update({k: v})
            table_data = table_data.mutate({k: None})
            schema_changed = True
        else:
            # print(f"Column {k} already exists in the schema.")
            if merged_schema[k] != v:
                print(f"[Database] Warning: Type mismatch for column {k}. Expected {merged_schema[k]}, got {v}.")
    return schema_changed, merged_schema, table_data

def insert_data(con, table_name, data):
    # Convert data to the appropriate format for insertion
    data_batch = {}
    if isinstance(data, dict):
        data = [data]
    for data_i in data:
        for key, value in data_i.items():
            data_batch.update({k: [] for k,v in to_ibis_type(key, value)})
    for data_i in data:
        ibis_value = {}
        for key, value in data_i.items():
            ibis_value.update(to_ibis_value(key, data_i[key]))
        for k,v in data_batch.items():
            if k in ibis_value:
                data_batch[k].append(ibis_value[k])
            else:
                data_batch[k].append(None)
    # check if there is new column
    current_schema = con.table(table_name).schema()
    for k,v in data_batch.items():
        if k not in current_schema:
            schema = generate_schema(data)
            schema_changed, new_schema, old_data_batch = update_schema(con, table_name, schema)
            if schema_changed:
                con.create_table(table_name, obj=old_data_batch, schema=new_schema, overwrite=True)
                try:
                    con.raw_sql("COMMIT;")
                    con.raw_sql("BEGIN TRANSACTION;")
                except Exception:
                    pass
    # insert data
    try:
        con.insert(table_name, data_batch)
        # print(f"[Database] Inserted data in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"[Database] Error: {e}")

class DB:
    def __init__(self, *args, **kwargs):
        self.con = connect_db(*args, **kwargs)

    def __enter__(self):
        return self.con

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"[Database] Closing connection")
        self.con.disconnect()
        print(f"[Database] Connection closed")
        if exc_type is not None:
            print(f"Exception: {exc_value}")
            return False
        return True

def connect_db(path, timeout=60, read_only=False, retry_interval=0.1, database='duckdb'):
    start_time = time.time()
    retry_count = 0
    while True:
        try:
            if database == 'sqlite':
                con = ibis.sqlite.connect(path)
            elif database == 'duckdb':
                con = ibis.connect(f"duckdb://{path}", read_only=read_only)
                con.raw_sql("SET enable_progress_bar=false")
            elif database == 'postgres':
                con = ibis.postgres.connect(
                    user="postgres",
                    password="test",
                    host="localhost",
                    port=5888,
                    database="test"
                )
            return con
        except Exception as e:
            if not "Could not set lock" in str(e):
                raise
            if retry_count%(1./retry_interval) == 0:
                print(f"[database] Waiting for connection to {path}...")
            if time.time() - start_time > timeout:
                raise TimeoutError("Connection timed out") from e
            time.sleep(retry_interval)
            retry_count += 1


def analyze_data_size(data, prefix="", threshold=1):
    for key, value in data.items():
        data_size = len(pickle.dumps(value))/1024/1024
        if data_size > threshold:
            print(f"[analyze_data_size] {prefix}{key}: {data_size:.2f} MB")
        if isinstance(value, dict):
            analyze_data_size(value, f"{prefix}['{key}']")
        if isinstance(value, list) and len(value) > 0:
            analyze_data_size({'0':value[0]}, f"{prefix}['{key}']")

"""Loads the nine sample documents from sample_data/ (see DATA_MODEL.md) into your database.
Usage:  python sample_seed.py            (reads MONGO_URI / MONGO_DB from the environment or .env)

Equivalent with the MongoDB tools:
  mongoimport --uri "$MONGO_URI" --db "$MONGO_DB" -c employees       --jsonArray --drop --file sample_data/employees.json
  mongoimport --uri "$MONGO_URI" --db "$MONGO_DB" -c attendance_logs --jsonArray --drop --file sample_data/attendance_logs.json

The samples show document SHAPES only. The grader uses a much larger hidden dataset with cases they do not show.
Passing on this sample does not mean your analytics are correct - re-read the business rules."""
import os
import pathlib

from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

project_root = pathlib.Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

here = project_root / "sample_data"
opts = json_util.JSONOptions(tz_aware=True)
configured_uri = os.getenv("MONGO_URI")
local_uri = "mongodb://localhost:27017"
uri_candidates = [configured_uri, local_uri] if configured_uri else [local_uri]
uri_candidates = list(dict.fromkeys(uri_candidates))
db_name = os.getenv("MONGO_DB") or "attendance_db_inter"

client = None
uri = None
connection_errors = []
for candidate in uri_candidates:
    try:
        client = MongoClient(candidate, tz_aware=True, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        uri = candidate
        break
    except PyMongoError as exc:
        endpoint = "configured MONGO_URI" if candidate == configured_uri else "local MongoDB"
        connection_errors.append(f"{endpoint}: {exc}")
        continue

if client is None or uri is None:
    raise RuntimeError(
        "Could not connect to MongoDB. Connection attempts failed:\n"
        + "\n".join(connection_errors)
        + "\nSet MONGO_URI to a valid MongoDB connection string or start a local "
        "MongoDB instance, for example: docker run -p 27017:27017 mongo:7"
    )

db = client[db_name]

for name in ("employees", "attendance_logs"):
    docs = json_util.loads((here / f"{name}.json").read_text(), json_options=opts)
    db[name].delete_many({})
    db[name].insert_many(docs)
    print(f"loaded {len(docs):>2} documents into '{db.name}.{name}'")

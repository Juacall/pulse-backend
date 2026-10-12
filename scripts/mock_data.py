"""
scripts/mock_data.py

Mock datasets for testing database creation, ingestion, and validation workflows
without calling external Congress APIs or modifying the production DuckDB database.
"""

from pathlib import Path
import sys
import duckdb

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from database import create_tables
except ImportError:
    from scripts.database import create_tables

# Mock district records: (state_abbr, district_num, geoid_cd, geoid_county)
MOCK_DISTRICTS = [
    ("GA", 14, "1314", "13115"),  # Floyd County, GA
    ("TX", 21, "4821", "48453"),  # Travis County, TX
]

# Mock House member records returned from API
MOCK_MEMBERS = [
    {
        "bioguideId": "G000596",
        "name": "Greene, Marjorie Taylor",
        "state": "GA",
        "chamber": "House",
        "district": 14,
        "partyName": "Republican",
        "_geoid_cd": "1314",
        "updateDate": "2025-01-15T12:00:00Z",
    },
    {
        "bioguideId": "R000614",
        "name": "Roy, Chip",
        "state": "TX",
        "chamber": "House",
        "district": 21,
        "partyName": "Republican",
        "_geoid_cd": "4821",
        "updateDate": "2025-01-16T15:30:00Z",
    },
]

# Mock Senate member records
MOCK_SENATE_MEMBERS = [
    {
        "bioguideId": "C001098",
        "name": "Cruz, Ted",
        "state": "TX",
        "chamber": "Senate",
        "district": None,
        "partyName": "Republican",
        "_geoid_cd": None,
        "updateDate": "2025-01-10T10:00:00Z",
    },
    {
        "bioguideId": "O000174",
        "name": "Ossoff, Jon",
        "state": "GA",
        "chamber": "Senate",
        "district": None,
        "partyName": "Democrat",
        "_geoid_cd": None,
        "updateDate": "2025-01-11T11:00:00Z",
    },
]

# Mock legislation records for member G000596 (House)
MOCK_SPONSORED_BILLS = [
    {
        "congress": 119,
        "type": "HR",
        "number": "101",
        "title": "Protect American Energy Act",
        "introducedDate": "2025-01-03",
        "updateDate": "2025-02-01T10:00:00Z",
        "latestAction": {
            "actionDate": "2025-02-01",
            "text": "Referred to the House Committee on Energy and Commerce.",
        },
    }
]

MOCK_COSPONSORED_BILLS = [
    {
        "congress": 119,
        "type": "HR",
        "number": "102",
        "title": "Border Security Acceleration Act",
        "introducedDate": "2025-01-04",
        "updateDate": "2025-02-02T11:00:00Z",
        "latestAction": {
            "actionDate": "2025-02-02",
            "text": "Became Public Law No: 119-1.",
        },
    }
]

# Mock legislation records for Senate member C001098
MOCK_SENATE_SPONSORED_BILLS = [
    {
        "congress": 119,
        "type": "S",
        "number": "201",
        "title": "American Energy Independence Act",
        "updateDate": "2025-02-05T10:00:00Z",
        "latestAction": {
            "actionDate": "2025-02-05",
            "text": "Read twice and referred to the Committee on Energy and Natural Resources.",
        },
    }
]

MOCK_SENATE_COSPONSORED_BILLS = [
    {
        "congress": 119,
        "type": "S",
        "number": "202",
        "title": "Federal Regulatory Reduction Act",
        "updateDate": "2025-02-06T11:00:00Z",
        "latestAction": {
            "actionDate": "2025-02-06",
            "text": "Read twice and referred to the Committee on Homeland Security and Governmental Affairs.",
        },
    }
]

# Mock amendment records
MOCK_AMENDMENTS = [
    {
        "number": "1",
        "type": "HAMDT",
        "description": "An amendment to specify federal funding limits.",
        "purpose": "Clarifies funding allocations in Section 3.",
        "sponsor": {"bioguideId": "G000596"},
        "updateDate": "2025-02-03T09:00:00Z",
    }
]

MOCK_SENATE_AMENDMENTS = [
    {
        "number": "10",
        "type": "SAMDT",
        "description": "Senate amendment to propose energy development offsets.",
        "purpose": "Provides regulatory flexibility for state projects.",
        "sponsor": {"bioguideId": "C001098"},
        "updateDate": "2025-02-07T14:00:00Z",
    }
]


def create_in_memory_db_with_seeds() -> duckdb.DuckDBPyConnection:
    """
    Creates an isolated in-memory DuckDB instance populated with mock seed tables
    and initializes the raw destination tables. Useful for unit and integration testing.
    """
    db = duckdb.connect(":memory:")

    # Create and populate target_counties mock seed
    db.execute("""
        CREATE TABLE target_counties (\n            state_fips INTEGER,\n            county_fips INTEGER\n        );\n        INSERT INTO target_counties VALUES\n            (13, 115),  -- GA Floyd\n            (48, 453);  -- TX Travis\n    """)

    # Create and populate census mock seed
    db.execute("""
        CREATE TABLE raw_census__cd11920_county20 (\n            GEOID_CD119_20 VARCHAR,\n            GEOID_COUNTY_20 VARCHAR\n        );\n        INSERT INTO raw_census__cd11920_county20 VALUES\n            ('1314', '13115'),\n            ('4821', '48453');\n    """)

    # Initialize raw target tables
    create_tables(db, replace=False)

    return db

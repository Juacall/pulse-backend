"""
scripts/ingestion.py

Data loading and idempotent upsert functions for DuckDB raw tables.
Includes MD5 checksum generation and timestamp tracking.
"""

import hashlib
from datetime import datetime, timezone


def _compute_hash(*values) -> str:
    """Compute an MD5 hash over given string/numeric values."""
    payload = "|".join("" if v is None else str(v) for v in values)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def load_members(db, members):
    """
    Idempotently upsert member rows into raw_members.
    Primary Key: (bioguide_id).

    Returns a stats dict describing the incremental outcome:
      {"total", "inserted", "updated", "unchanged"}
    - inserted:  bioguide_id not previously in raw_members
    - updated:   existed, but row_hash changed (content differs)
    - unchanged: existed with identical row_hash (write skipped by the ON CONFLICT WHERE clause)
    """
    if not members:
        return {"total": 0, "inserted": 0, "updated": 0, "unchanged": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for m in members:
        bid = m.get("bioguideId")
        name = m.get("name")
        state = m.get("state")
        chamber = m.get("chamber") or ("House" if m.get("district") is not None else "Senate")
        district = m.get("district")
        party = m.get("partyName")
        geoid_cd = m.get("_geoid_cd")
        update_date = m.get("updateDate")

        row_hash = _compute_hash(bid, name, state, chamber, district, party, geoid_cd)
        rows.append((bid, name, state, chamber, district, party, geoid_cd, update_date, row_hash, now_iso))

    # Classify each incoming row against what's already stored so the caller can
    # report new / changed / unchanged counts. Done before the upsert because the
    # ON CONFLICT WHERE clause silently skips unchanged rows (no affected-row signal).
    existing_hashes = {}
    bids = [r[0] for r in rows if r[0] is not None]
    if bids:
        placeholders = ",".join("?" for _ in bids)
        existing_hashes = {
            bid: h
            for bid, h in db.execute(
                f"SELECT bioguide_id, row_hash FROM main.raw_members WHERE bioguide_id IN ({placeholders})",
                bids,
            ).fetchall()
        }

    inserted = updated = unchanged = 0
    for r in rows:
        bid, new_hash = r[0], r[8]
        if bid not in existing_hashes:
            inserted += 1
        elif existing_hashes[bid] != new_hash:
            updated += 1
        else:
            unchanged += 1

    db.executemany("""
        INSERT INTO main.raw_members (
            bioguide_id, name, state, chamber, district, party, geoid_cd, update_date, row_hash, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (bioguide_id) DO UPDATE SET
            name = EXCLUDED.name,
            state = EXCLUDED.state,
            chamber = EXCLUDED.chamber,
            district = EXCLUDED.district,
            party = EXCLUDED.party,
            geoid_cd = EXCLUDED.geoid_cd,
            update_date = EXCLUDED.update_date,
            row_hash = EXCLUDED.row_hash,
            ingested_at = EXCLUDED.ingested_at
        WHERE main.raw_members.row_hash != EXCLUDED.row_hash
    """, rows)

    return {
        "total": len(rows),
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
    }


def load_bills(db, bills, member_id, relationship):
    """
    Idempotently upsert bill rows into raw_bills.
    Primary Key: (congress, bill_type, bill_number, member_id, relationship).

    Returns counts for inserted, updated, and unchanged member-bill relationships.
    """
    if not bills:
        return {"total": 0, "inserted": 0, "updated": 0, "unchanged": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for b in bills:
        congress = b.get("congress")
        bill_type = b.get("type")
        bill_number = str(b.get("number"))
        title = b.get("title")
        introduced_date = b.get("introducedDate")
        update_date = b.get("updateDate") or b.get("updateDateIncludingText")

        # Current bill status: the Congress API exposes the most-recent action
        # inline on the sponsored/cosponsored list response (no extra call). We
        # track current status only, so this is overwritten on each run.
        latest_action = b.get("latestAction") or {}
        latest_action_date = latest_action.get("actionDate")
        latest_action_text = latest_action.get("text")

        row_hash = _compute_hash(
            congress, bill_type, bill_number, title, member_id, relationship,
            introduced_date, latest_action_date, latest_action_text
        )
        rows.append((
            congress, bill_type, bill_number, title,
            introduced_date, latest_action_date, latest_action_text,
            member_id, relationship, update_date, row_hash, now_iso
        ))

    existing_hashes = {}
    chunk_size = 250
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        placeholders = ",".join("(?, ?, ?)" for _ in chunk)
        params = [member_id, relationship]
        for row in chunk:
            params.extend((row[0], row[1], row[2]))
        existing_hashes.update({
            (congress, bill_type, bill_number, existing_member_id, existing_relationship): row_hash
            for congress, bill_type, bill_number, existing_member_id, existing_relationship, row_hash
            in db.execute(
                f"""
                SELECT congress, bill_type, bill_number, member_id, relationship, row_hash
                FROM main.raw_bills
                WHERE member_id = ? AND relationship = ?
                  AND (congress, bill_type, bill_number) IN ({placeholders})
                """,
                params,
            ).fetchall()
        })

    inserted = updated = unchanged = 0
    for row in rows:
        key = (row[0], row[1], row[2], row[7], row[8])
        existing_hash = existing_hashes.get(key)
        if existing_hash is None:
            inserted += 1
        elif existing_hash != row[10]:
            updated += 1
        else:
            unchanged += 1

    db.executemany("""
        INSERT INTO main.raw_bills (
            congress, bill_type, bill_number, title, introduced_date,
            latest_action_date, latest_action_text, member_id, relationship,
            update_date, row_hash, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (congress, bill_type, bill_number, member_id, relationship) DO UPDATE SET
            title = EXCLUDED.title,
            introduced_date = EXCLUDED.introduced_date,
            latest_action_date = EXCLUDED.latest_action_date,
            latest_action_text = EXCLUDED.latest_action_text,
            update_date = EXCLUDED.update_date,
            row_hash = EXCLUDED.row_hash,
            ingested_at = EXCLUDED.ingested_at
        WHERE main.raw_bills.row_hash != EXCLUDED.row_hash
    """, rows)

    return {
        "total": len(rows),
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
    }


def load_amendments(db, amendments, congress, bill_type, bill_number):
    """
    Idempotently upsert amendment rows into raw_amendments for a parent bill.
    Primary Key: (congress, bill_type, bill_number, amendment_number).
    """
    if not amendments:
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for a in amendments:
        sponsor = a.get("sponsor")
        sponsor_id = sponsor.get("bioguideId") if isinstance(sponsor, dict) else None
        amendment_num = str(a.get("number", ""))
        amendment_type = a.get("type")
        description = a.get("description")
        purpose = a.get("purpose")
        update_date = a.get("updateDate")

        row_hash = _compute_hash(
            congress, bill_type, bill_number, amendment_num, amendment_type,
            description, purpose, sponsor_id
        )
        rows.append((
            congress, bill_type, bill_number, amendment_num, amendment_type,
            description, purpose, sponsor_id, update_date, row_hash, now_iso
        ))

    db.executemany("""
        INSERT INTO main.raw_amendments (
            congress, bill_type, bill_number, amendment_number, amendment_type,
            description, purpose, sponsor_id, update_date, row_hash, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (congress, bill_type, bill_number, amendment_number, amendment_type) DO UPDATE SET
            amendment_type = EXCLUDED.amendment_type,
            description = EXCLUDED.description,
            purpose = EXCLUDED.purpose,
            sponsor_id = EXCLUDED.sponsor_id,
            update_date = EXCLUDED.update_date,
            row_hash = EXCLUDED.row_hash,
            ingested_at = EXCLUDED.ingested_at
        WHERE main.raw_amendments.row_hash != EXCLUDED.row_hash
    """, rows)

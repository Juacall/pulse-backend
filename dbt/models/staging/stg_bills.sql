-- stg_bills.sql
--
-- Clean and rename raw_bills.
-- One row per member-bill relationship — a bill appears once per connected member.
-- member_relationship is either 'sponsor' or 'cosponsor'.
--
-- BILL STATUS (current status only):
--   latest_action_date / latest_action_text are the bill's most-recent action as
--   reported by the Congress API — a point-in-time snapshot overwritten on each
--   ingestion run (no status history is retained).
--   bill_status is a coarse, derived category for aggregation. It is a heuristic
--   over latest_action_text; always fall back to latest_action_text for detail.
{{ config(materialized = 'view') }}

select
    cast(congress as integer)               as congress,
    upper(trim(bill_type))                  as bill_type,
    trim(bill_number)                       as bill_number,
    upper(trim(bill_type))
        || '-' || trim(bill_number)         as bill_id,
    trim(title)                             as bill_title,
    trim(member_id)                         as member_id,
    lower(trim(relationship))               as member_relationship,  -- 'sponsor' or 'cosponsor'
    case upper(trim(bill_type))
        when 'HR'      then 'House Bill'
        when 'HRES'    then 'House Resolution'
        when 'HJRES'   then 'House Joint Resolution'
        when 'HCONRES' then 'House Concurrent Resolution'
        else upper(trim(bill_type))
    end                                     as bill_type_label,
    try_cast(introduced_date as date)       as bill_introduced_date,
    -- current status (point-in-time snapshot)
    try_cast(latest_action_date as date)    as bill_latest_action_date,
    nullif(trim(latest_action_text), '')    as bill_latest_action_text,
    -- derived coarse status bucket for filtering/aggregation
    case
        when latest_action_text is null or trim(latest_action_text) = ''
            then 'Unknown'
        when lower(latest_action_text) like '%became public law%'
          or lower(latest_action_text) like '%became private law%'
          or lower(latest_action_text) like '%signed by president%'
            then 'Became Law'
        when lower(latest_action_text) like '%vetoed%'
            then 'Vetoed'
        when lower(latest_action_text) like '%passed senate%'
          or lower(latest_action_text) like '%agreed to in senate%'
            then 'Passed Senate'
        when lower(latest_action_text) like '%passed house%'
          or lower(latest_action_text) like '%agreed to in house%'
          or lower(latest_action_text) like '%passed/agreed to in house%'
            then 'Passed House'
        when lower(latest_action_text) like '%referred to%'
          or lower(latest_action_text) like '%committee%'
            then 'In Committee'
        when lower(latest_action_text) like '%introduced%'
            then 'Introduced'
        else 'Other'
    end                                     as bill_status
from {{ source('congress', 'raw_bills') }}
where
    member_id is not null
    and trim(member_id) != ''
    and bill_type is not null
    and bill_number is not null
    and trim(bill_number) != ''
    and congress is not null
    and upper(trim(bill_type)) in (
        'HR', 'HRES', 'HJRES', 'HCONRES',
        'S', 'SRES', 'SJRES', 'SCONRES'
    )
    and lower(trim(relationship)) in ('sponsor', 'cosponsor')

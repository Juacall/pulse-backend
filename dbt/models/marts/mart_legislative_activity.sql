-- mart_legislative_activity.sql
--
-- GRAIN: one row per county x target_member x activity.
--
-- Filter by county_fips or county_name to see all legislative activity for
-- that county's House members.
--
-- county_fips is the full 5-digit FIPS (state + county) e.g. '13047'.
-- activity_id and activity_title are unified convenience fields — use these
-- for filtering, display, and aggregation rather than bill_id/amendment_id.
-- bill_title is the parent bill title for both bills and amendments.
-- amendment_title is null for bills.
-- bill_status / bill_latest_action_* reflect the CURRENT status of the parent
-- bill (point-in-time snapshot, no history). Amendments inherit their parent
-- bill's status. Use bill_status for aggregation, bill_latest_action_text for detail.
-- amendment_sponsor_*_if_target is only populated when the amendment sponsor
-- is themselves a target-district member.
--
-- RELATIONSHIP VALUES:
--   'sponsor'                           — member sponsored the bill
--   'cosponsor'                         — member cosponsored the bill
--   'amendment_sponsor'                 — member sponsored the amendment
--   'sponsored bill received amendment' — member's bill received this amendment
--
-- Use COUNT(DISTINCT activity_id) not COUNT(*) when aggregating — the same
-- activity appears once per county and once per member.
{{ config(materialized = 'table') }}
select
-- County (primary filter anchor)
    mdc.county_fips,                            -- full 5-digit FIPS e.g. '13047'
    mdc.county_name,
    mdc.county_state,
-- Target-district member connected to this activity via bill ownership
    mdc.member_id                               as target_member_id,
    mdc.member_name                             as target_member_name,
    mdc.chamber                                 as target_member_chamber,
    mdc.state_name                              as target_member_state,
    mdc.state_code                              as target_member_state_code,
    mdc.state_name                              as target_member_state_name,
    mdc.district_number                         as target_member_district,
    mdc.party_name                              as target_member_party,
-- How the member relates to the activity
    la.relationship,
-- Activity (unified fields first, then underlying columns)
    la.activity_type,                           -- 'bill' or 'amendment'
    coalesce(la.amendment_id, la.bill_id)       as activity_id,
    coalesce(la.amendment_title, la.bill_title) as activity_title,
    la.bill_id,
    la.bill_title,                              -- parent bill title for both bills and amendments
-- Current status of the (parent) bill — snapshot, no history
    la.bill_status,                            -- coarse category for aggregation
    la.bill_introduced_date,
    la.bill_latest_action_date,                -- date of the most-recent action
    la.bill_latest_action_text,                -- full text of the most-recent action
    la.amendment_id,
    la.amendment_title,                         -- null for bills
-- Who proposed the amendment (null for bills, null if outside target districts)
    la.amendment_sponsor_id,
    amend_sponsor.chamber                       as amendment_sponsor_chamber_if_target,
    amend_sponsor.member_name                   as amendment_sponsor_name_if_target,
    amend_sponsor.state_code                    as amendment_sponsor_state_if_target,
    amend_sponsor.party_name                    as amendment_sponsor_party_if_target
from {{ ref('int_legislative_activity') }}              as la
inner join {{ ref('int_members_districts_counties') }}  as mdc
on la.member_id = mdc.member_id
left join {{ ref('stg_members') }}                      as amend_sponsor
on la.amendment_sponsor_id = amend_sponsor.member_id
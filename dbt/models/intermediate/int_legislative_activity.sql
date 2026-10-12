-- int_legislative_activity.sql
--
-- GRAIN: one row per member x activity.
--
-- Unions bills and amendments into a single activity feed, each row tied to a
-- target district member. is sent into the mart after joining with member-county data.
--
-- Bills are linked to the member who sponsored or cosponsored them.
--
-- Amendments link via their parent bill — so each amendment fans out to one row
-- per member connected to that bill. The relationship column captures whether
-- the member personally sponsored the amendment or just owned the parent bill.
--
-- bill_title is always the parent bill title. amendment_title is null for bills.
--
-- member_id is always the target-district member. amendment_sponsor_id is whoever
-- proposed the amendment. If the amendment sponsor isn't a
-- target district member, their name won't appear; that's intentional since the
-- grain is target district member x activity, not amendment sponsor x activity.
--
-- RELATIONSHIP VALUES:
--   'sponsor'                           — member sponsored the bill
--   'cosponsor'                         — member cosponsored the bill
--   'amendment_sponsor'                 — member sponsored the amendment
--   'sponsored bill received amendment' — member's bill received this amendment
{{ config(materialized = 'table') }}

with bills as (
    select
        member_id,
        bill_id                             as bill_id,
        bill_title                          as bill_title,
        cast(null as varchar)               as amendment_id,
        cast(null as varchar)               as amendment_title,
        cast(null as varchar)               as amendment_sponsor_id,
        'bill'                              as activity_type,
        member_relationship                 as relationship,
        congress,
        bill_type,
        bill_number,
        bill_status                         as bill_status,
        bill_introduced_date                as bill_introduced_date,
        bill_latest_action_date             as bill_latest_action_date,
        bill_latest_action_text             as bill_latest_action_text
    from {{ ref('stg_bills') }}
),

amendments as (
    select
        b.member_id,
        b.bill_id                           as bill_id,
        b.bill_title                        as bill_title,
        a.amendment_id                      as amendment_id,
        a.amendment_title                   as amendment_title,
        a.sponsor_id                        as amendment_sponsor_id,
        'amendment'                         as activity_type,
        case
            when a.sponsor_id = b.member_id then 'amendment_sponsor'
            else 'sponsored bill received amendment'
        end                                 as relationship,
        a.congress,
        a.bill_type,
        a.bill_number,
        b.bill_status                       as bill_status,
        b.bill_introduced_date              as bill_introduced_date,
        b.bill_latest_action_date           as bill_latest_action_date,
        b.bill_latest_action_text           as bill_latest_action_text
    from {{ ref('stg_amendments') }} as a
    inner join {{ ref('stg_bills') }} as b
        on  a.congress    = b.congress
        and a.bill_type   = b.bill_type
        and a.bill_number = b.bill_number
),

unmerged as (
    select * from bills
    union all
    select * from amendments
)

select
    -- Generate primary surrogate key on unmerged result
    md5(
            coalesce(cast(member_id as varchar), '') || '-' ||
            coalesce(cast(relationship as varchar), '') || '-' ||
            coalesce(cast(activity_type as varchar), '') || '-' ||
            coalesce(cast(coalesce(amendment_id, bill_id) as varchar), '')
    ) as legislative_activity_sk,

    member_id,
    bill_id,
    bill_title,
    amendment_id,
    amendment_title,
    amendment_sponsor_id,
    activity_type,
    relationship,
    congress,
    bill_type,
    bill_number,
    bill_status,
    bill_introduced_date,
    bill_latest_action_date,
    bill_latest_action_text
from unmerged
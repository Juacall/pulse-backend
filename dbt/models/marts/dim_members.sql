{{ config(materialized = 'table') }}

with member_activity as (
    select
        target_member_id as bioguide_id,

        -- Total Sponsored Bills
        count(distinct case
            when relationship = 'sponsor' and activity_type = 'bill'
            then bill_id
        end) as total_sponsored_bills,

        -- Total Cosponsored Bills
        count(distinct case
            when relationship = 'cosponsor' and activity_type = 'bill'
            then bill_id
        end) as total_cosponsored_bills,

        -- Enacted Bills
        count(distinct case
            when relationship = 'sponsor'
             and activity_type = 'bill'
             and (
                 bill_status = 'Became Law'
                 or bill_latest_action_text ilike '%became public law%'
                 or bill_latest_action_text ilike '%became private law%'
                 or bill_latest_action_text ilike '%signed by president%'
             )
            then bill_id
        end) as total_enacted_bills

    from {{ ref('mart_legislative_activity') }}
    group by target_member_id
)

select
    members.member_id as bioguide_id,
    members.member_name as name,
    members.state_code as state,
    members.chamber,
    members.party_name as party,
    coalesce(activity.total_sponsored_bills, 0) as sponsored_count,
    coalesce(activity.total_cosponsored_bills, 0) as cosponsored_count,
    coalesce(activity.total_enacted_bills, 0) as enacted_count,

    -- Calculated Pass Rate at 1-row-per-member grain
    case
        when coalesce(activity.total_sponsored_bills, 0) > 0
            then round(
                activity.total_enacted_bills * 100.0
                / activity.total_sponsored_bills,
                1
            )
        else cast(null as double)
    end as pass_rate
from {{ ref('stg_members') }} as members
left join member_activity as activity
    on members.member_id = activity.bioguide_id
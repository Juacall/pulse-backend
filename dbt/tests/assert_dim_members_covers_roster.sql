select
    members.member_id
from {{ ref('stg_members') }} as members
left join {{ ref('dim_members') }} as dimension
    on members.member_id = dimension.bioguide_id
where dimension.bioguide_id is null

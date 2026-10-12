select
    bioguide_id
from {{ ref('dim_members') }}
where
    (
        sponsored_count = 0
        and (pass_rate is not null or enacted_count <> 0)
    )
    or (
        sponsored_count > 0
        and pass_rate is distinct from round(
            enacted_count * 100.0 / sponsored_count,
            1
        )
    )
    or enacted_count > sponsored_count

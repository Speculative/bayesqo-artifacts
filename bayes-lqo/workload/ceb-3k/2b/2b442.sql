SELECT COUNT(*) FROM title as t,
kind_type as kt,
info_type as it1,
movie_info as mi1,
movie_info as mi2,
info_type as it2,
cast_info as ci,
role_type as rt,
name as n,
movie_keyword as mk,
keyword as k
WHERE
t.id = ci.movie_id
AND t.id = mi1.movie_id
AND t.id = mi2.movie_id
AND t.id = mk.movie_id
AND k.id = mk.keyword_id
AND mi1.movie_id = mi2.movie_id
AND mi1.info_type_id = it1.id
AND mi2.info_type_id = it2.id
AND (it1.id in ('8'))
AND (it2.id in ('105'))
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.role_id = rt.id
AND (mi1.info in ('USA'))
AND (mi2.info in ('$1,000,000','$10,000','$100','$100,000','$15,000','$2,000','$20,000','$25,000','$250,000','$3,000','$30,000','$5,000','$50,000','$500,000'))
AND (kt.kind in ('tv movie','tv series','video game'))
AND (rt.role in ('cinematographer','editor'))
AND (n.gender in ('f','m'))
AND (t.production_year <= 2015)
AND (t.production_year >= 1975)
AND (k.keyword IN ('anal-sex','bare-chested-male','dancing','doctor','father-son-relationship','homosexual','lesbian-sex','number-in-title'))

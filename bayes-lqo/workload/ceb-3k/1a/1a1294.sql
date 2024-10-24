SELECT COUNT(*) FROM title as t,
kind_type as kt,
movie_info as mi1,
info_type as it1,
movie_info as mi2,
info_type as it2,
cast_info as ci,
role_type as rt,
name as n
WHERE
t.id = ci.movie_id
AND t.id = mi1.movie_id
AND t.id = mi2.movie_id
AND mi1.movie_id = mi2.movie_id
AND mi1.info_type_id = it1.id
AND mi2.info_type_id = it2.id
AND it1.id = '3'
AND it2.id = '7'
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.role_id = rt.id
AND mi1.info IN ('Adventure','Crime','Drama','Family','Horror','Short')
AND mi2.info IN ('CAM:Red One Camera','OFM:35 mm','PCS:Super 35','PFM:35 mm','RAT:1.37 : 1','RAT:1.78 : 1','RAT:2.35 : 1')
AND kt.kind IN ('episode','movie','video movie')
AND rt.role IN ('actor','composer')
AND n.gender IN ('m')
AND t.production_year <= 2015
AND 1975 < t.production_year

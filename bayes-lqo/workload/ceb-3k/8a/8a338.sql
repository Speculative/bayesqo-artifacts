SELECT COUNT(*) FROM title as t,
kind_type as kt,
info_type as it1,
movie_info as mi1,
cast_info as ci,
role_type as rt,
name as n,
movie_keyword as mk,
keyword as k,
movie_companies as mc,
company_type as ct,
company_name as cn
WHERE
t.id = ci.movie_id
AND t.id = mc.movie_id
AND t.id = mi1.movie_id
AND t.id = mk.movie_id
AND mc.company_type_id = ct.id
AND mc.company_id = cn.id
AND k.id = mk.keyword_id
AND mi1.info_type_id = it1.id
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.role_id = rt.id
AND (it1.id IN ('7'))
AND (mi1.info in ('RAT:1.33 : 1','RAT:1.78 : 1'))
AND (kt.kind in ('episode','tv series'))
AND (rt.role in ('actor','actress','miscellaneous crew'))
AND (n.gender in ('f','m') OR n.gender IS NULL)
AND (n.surname_pcode in ('A45','B65','C4','C65','F652','G6','H52','J52','L2','L5','L52','M24','P6','W3'))
AND (t.production_year <= 1975)
AND (t.production_year >= 1925)
AND (cn.name in ('Columbia Broadcasting System (CBS)','Metro-Goldwyn-Mayer (MGM)','Warner Home Video'))
AND (ct.kind in ('distributors','production companies'))

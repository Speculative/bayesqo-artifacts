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
AND (mi1.info in ('CAM:Panavision Cameras and Lenses','LAB:DeLuxe, Hollywood (CA), USA','LAB:Technicolor','LAB:Technicolor, Hollywood (CA), USA','MET:300 m','MET:600 m','OFM:35 mm','PCS:Spherical','PFM:35 mm','RAT:1.20 : 1','RAT:1.33 : 1','RAT:1.78 : 1','RAT:16:9 HD'))
AND (kt.kind in ('episode','movie','tv movie','tv series'))
AND (rt.role in ('actress','producer'))
AND (n.gender in ('f'))
AND (n.surname_pcode in ('B62','B65','H4','J52','L','M6','S53') OR n.surname_pcode IS NULL)
AND (t.production_year <= 2015)
AND (t.production_year >= 1925)
AND (cn.name in ('Fox Network','Independent Television (ITV)','National Broadcasting Company (NBC)','Paramount Pictures','Sony Pictures Home Entertainment','Universal Pictures','Universal TV','Warner Bros'))
AND (ct.kind in ('distributors','production companies'))

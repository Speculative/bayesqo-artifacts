SELECT COUNT(*) FROM title as t,
kind_type as kt,
info_type as it1,
movie_info as mi1,
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
AND (it1.id in ('17'))
AND (it2.id in ('4'))
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.role_id = rt.id
AND (mi1.info IN ('Last show of the series.','One of over 700 Paramount Productions, filmed between 1929 and 1949, which were sold to MCA/Universal in 1958 for television distribution, and have been owned and controlled by Universal ever since.','Original French title is undetermined.'))
AND (mi2.info IN ('English','French','German','Italian','Japanese','Latin','Spanish'))
AND (kt.kind in ('episode','movie','tv movie','video game','video movie'))
AND (rt.role in ('actor','actress','costume designer','producer','writer'))
AND (n.gender IN ('m'))
AND (t.production_year <= 1975)
AND (t.production_year >= 1875)
AND (t.title in ('(#3.24)','(#3.39)','April Fool','Arrival','As the World Turns','Ave Maria','Betrayed','Contraband','Excuse My Dust','Festival','Goin to Town','Lets Dance','Marie Antoinette','Meet the People','Monkey Business','My Favorite Spy','Ordeal','Outcast','Random Harvest','Robin Hood','San Francisco','Secrets','Stranded','The Baby Sitter','The Best Years of Our Lives','The Big Show','The Engagement','The Morning After','The Other Man','The Prisoner of Zenda','The Proposal','The Rainmaker','The Road Back','The Star','Zaza'))

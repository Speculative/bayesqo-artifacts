SELECT COUNT(*)
FROM
site as s,
so_user as u1,
tag as t1,
tag_question as tq1,
question as q1,
badge as b1,
account as acc
WHERE
s.site_id = u1.site_id
AND s.site_id = b1.site_id
AND s.site_id = t1.site_id
AND s.site_id = tq1.site_id
AND s.site_id = q1.site_id
AND t1.id = tq1.tag_id
AND q1.id = tq1.question_id
AND q1.owner_user_id = u1.id
AND acc.id = u1.account_id
AND b1.user_id = u1.id
AND (q1.score >= 0)
AND (q1.score <= 5)
AND s.site_name = 'stackoverflow'
AND (t1.name in ('clion','cocos2d-iphone','e4','lex','mod-wsgi','parent-child'))
AND (acc.website_url ILIKE ('%org'))

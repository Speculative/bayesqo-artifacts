from datetime import date, timedelta
from typing import NamedTuple, Union

from dateutil.relativedelta import relativedelta


def generate_shift_sql(target_date: date) -> str:
    sql = """
    -- Configure indexes
    CREATE INDEX IF NOT EXISTS account_id_idx ON account (id);
    CREATE INDEX IF NOT EXISTS so_user_site_id_idx ON so_user (site_id);
    CREATE INDEX IF NOT EXISTS so_user_site_id_id_idx ON so_user (site_id, id);
    CREATE INDEX IF NOT EXISTS so_user_id_idx ON so_user (id);
    CREATE INDEX IF NOT EXISTS so_user_account_id_idx ON so_user (account_id);
    CREATE INDEX IF NOT EXISTS badge_site_id_idx ON badge (site_id);
    CREATE INDEX IF NOT EXISTS badge_site_id_user_id_idx ON badge (site_id, user_id);
    CREATE INDEX IF NOT EXISTS badge_user_id_idx ON badge (user_id);
    CREATE INDEX IF NOT EXISTS tag_site_id_idx ON tag (site_id);
    CREATE INDEX IF NOT EXISTS tag_site_id_id_idx ON tag (site_id, id);
    CREATE INDEX IF NOT EXISTS tag_id_idx ON tag (id);
    CREATE INDEX IF NOT EXISTS tag_question_tag_id_idx ON tag_question (tag_id);
    CREATE INDEX IF NOT EXISTS tag_question_site_id_tag_id_idx ON tag_question (site_id, tag_id);
    CREATE INDEX IF NOT EXISTS tag_question_question_id_idx ON tag_question (question_id);
    CREATE INDEX IF NOT EXISTS tag_question_site_id_idx ON tag_question (site_id);
    CREATE INDEX IF NOT EXISTS tag_question_site_id_question_id_idx ON tag_question (site_id, question_id);
    CREATE INDEX IF NOT EXISTS answer_owner_user_id_idx ON answer (owner_user_id);
    CREATE INDEX IF NOT EXISTS answer_question_id_idx ON answer (question_id);
    CREATE INDEX IF NOT EXISTS answer_site_id_idx ON answer (site_id);
    CREATE INDEX IF NOT EXISTS answer_site_id_question_id_idx ON answer (site_id, question_id);
    CREATE INDEX IF NOT EXISTS answer_site_id_owner_user_id_idx ON answer (site_id, owner_user_id);
    CREATE INDEX IF NOT EXISTS question_id_idx ON question (id);
    CREATE INDEX IF NOT EXISTS question_owner_user_id_idx ON question (owner_user_id);
    CREATE INDEX IF NOT EXISTS question_site_id_id_idx ON question (site_id, id);
    CREATE INDEX IF NOT EXISTS question_site_id_idx ON question (site_id);
    CREATE INDEX IF NOT EXISTS question_site_id_owner_user_id_idx ON question (site_id, owner_user_id);
    CREATE INDEX IF NOT EXISTS site_site_id_idx ON site (site_id);
    CREATE INDEX IF NOT EXISTS comment_site_id_post_id_idx ON comment (site_id, post_id);
    CREATE INDEX IF NOT EXISTS post_link_site_id_idx ON post_link (site_id);
    CREATE INDEX IF NOT EXISTS post_link_site_id_post_id_from_idx ON post_link (site_id, post_id_from);
    CREATE INDEX IF NOT EXISTS post_link_site_id_post_id_to_idx ON post_link (site_id, post_id_to);
    
    CREATE INDEX IF NOT EXISTS answer_creation_date_idx ON answer(creation_date);
    CREATE INDEX IF NOT EXISTS badge_date_idx ON badge(date);
    CREATE INDEX IF NOT EXISTS comment_date_idx ON comment(date);
    CREATE INDEX IF NOT EXISTS post_link_date_idx ON post_link(date);
    
    DELETE FROM answer WHERE creation_date > '{0}';
    DELETE FROM badge WHERE date > '{0}';
    DELETE FROM comment WHERE date > '{0}';
    DELETE FROM post_link WHERE date > '{0}';
    
    DROP INDEX answer_creation_date_idx;
    DROP INDEX badge_date_idx;
    DROP INDEX comment_date_idx;
    DROP INDEX post_link_date_idx;
    
    -- question
    CREATE INDEX IF NOT EXISTS question_creation_date_site_id_id_idx ON question(creation_date, site_id, id);
    DELETE FROM answer
    	USING question
    	WHERE answer.site_id = question.site_id AND
    				answer.question_id = question.id AND
    				question.creation_date > '{0}';
    DELETE FROM post_link
    	USING question
    	WHERE post_link.site_id = question.site_id AND
    				post_link.post_id_to = question.id AND
    				question.creation_date > '{0}';
    DELETE FROM post_link
    	USING question
    	WHERE post_link.site_id = question.site_id AND
    				post_link.post_id_from = question.id AND
    				question.creation_date > '{0}';
    DELETE FROM tag_question
    	USING question
    	WHERE tag_question.site_id = question.site_id AND
    				tag_question.question_id = question.id AND
    				question.creation_date > '{0}';
    
    DELETE FROM question WHERE creation_date > '{0}';
    DROP INDEX question_creation_date_site_id_id_idx;
    
    -- so_user
    CREATE INDEX IF NOT EXISTS so_user_creation_date_site_id_id_idx ON so_user(creation_date, site_id, id);
    
    -- For direct constraints on so_user
    DELETE FROM answer
    	USING so_user
    	WHERE answer.site_id = so_user.site_id AND
    				answer.owner_user_id = so_user.id AND
    				so_user.creation_date > '{0}';
    DELETE FROM answer
    	USING so_user
    	WHERE answer.site_id = so_user.site_id AND
    				answer.last_editor_id = so_user.id AND
    				so_user.creation_date > '{0}';
    DELETE FROM badge
    	USING so_user
    	WHERE badge.site_id = so_user.site_id AND
    				badge.user_id = so_user.id AND
    				so_user.creation_date > '{0}';
    
    -- For transitive constraints on so_user through question
    DELETE FROM answer
    	USING question, so_user
    	WHERE answer.site_id = question.site_id AND
    				answer.question_id = question.id AND
    				question.site_id = so_user.site_id AND
    				question.owner_user_id = so_user.id AND
    				so_user.creation_date > '{0}';
    DELETE FROM answer
    	USING question, so_user
    	WHERE answer.site_id = question.site_id AND
    				answer.question_id = question.id AND
    				question.site_id = so_user.site_id AND
    				question.last_editor_id = so_user.id AND
    				so_user.creation_date > '{0}';
    				
    DELETE FROM post_link
    	USING question, so_user
    	WHERE post_link.site_id = question.site_id AND
    				post_link.post_id_to = question.id AND
    				question.site_id = so_user.site_id AND
    				question.owner_user_id = so_user.id AND
    				so_user.creation_date > '{0}';
    DELETE FROM post_link
    	USING question, so_user
    	WHERE post_link.site_id = question.site_id AND
    				post_link.post_id_to = question.id AND
    				question.site_id = so_user.site_id AND
    				question.last_editor_id = so_user.id AND
    				so_user.creation_date > '{0}';
    DELETE FROM post_link
    	USING question, so_user
    	WHERE post_link.site_id = question.site_id AND
    				post_link.post_id_from = question.id AND
    				question.site_id = so_user.site_id AND
    				question.owner_user_id = so_user.id AND
    				so_user.creation_date > '{0}';
    DELETE FROM post_link
    	USING question, so_user
    	WHERE post_link.site_id = question.site_id AND
    				post_link.post_id_from = question.id AND
    				question.site_id = so_user.site_id AND
    				question.last_editor_id = so_user.id AND
    				so_user.creation_date > '{0}';
    
    -- The remaining direct constraints on so_user from question
    DELETE FROM question
    	USING so_user
    	WHERE question.site_id = so_user.site_id AND
    				question.owner_user_id = so_user.id AND
    				so_user.creation_date > '{0}';
    DELETE FROM question
    	USING so_user
    	WHERE question.site_id = so_user.site_id AND
    				question.last_editor_id = so_user.id AND
    				so_user.creation_date > '{0}';
    
    				
    CREATE INDEX IF NOT EXISTS so_user_creation_date_idx ON so_user(creation_date);
    CREATE INDEX IF NOT EXISTS answer_site_id_last_editor_id_idx ON answer(site_id, last_editor_id);
    CREATE INDEX IF NOT EXISTS question_site_id_last_editor_id_idx ON question(site_id, last_editor_id);
    DELETE FROM so_user WHERE creation_date > '{0}';
    DROP INDEX so_user_creation_date_site_id_id_idx;
    DROP INDEX so_user_creation_date_idx;
    DROP INDEX answer_site_id_last_editor_id_idx;
    DROP INDEX question_site_id_last_editor_id_idx;
    
    DELETE FROM account
    	WHERE id IN (
    		SELECT account.id
    		FROM account
    		LEFT JOIN so_user
    		ON so_user.account_id = account.id
    		WHERE so_user.account_id IS NULL
    	);
    
    DELETE from tag
    	WHERE id in (
    		SELECT tag.id
    		FROM tag
    		LEFT JOIN tag_question
    		ON tag_question.tag_id = tag.id
    		WHERE tag_question.tag_id IS NULL
    );
    """.format(
        target_date
    )
    return sql


# Version without using dateutil if you don't have it installed
# Months have to be expressed as number of days or weeks
# shift_by = timedelta(
#     days=1
#     # weeks=1
# )
# target_date = (MAX_DATE - shift_by).isoformat()

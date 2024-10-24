DROP INDEX IF EXISTS answer_creation_date_idx,
	answer_last_editor_id_idx,
	comment_site_id_user_id_idx,
	question_creation_date_idx,
	question_last_editor_id_idx,
	so_user_creation_date_idx,
	so_user_last_access_date_idx,
	tag_question_site_id_tag_id_question_id_idx,
	post_link_site_id_post_id_from_post_id_to_idx;

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

# stop and remove pg
systemctl stop postgresql
rm -rf /var/lib/postgres/data
mkdir -p /var/lib/postgres/data
chown -R postgres /var/lib/postgres

# create a new DB and user
su postgres -c 'initdb -D /var/lib/postgres/data'
systemctl enable postgresql
systemctl start postgresql
su postgres -c 'createuser -s imdb'
su postgres -c 'createdb imdb'

# make sure it is working
psql -U imdb -h localhost -c 'select 1;'

# load the data dump
pg_restore -U imdb -h localhost -d imdb -c /root/imdb_pg11


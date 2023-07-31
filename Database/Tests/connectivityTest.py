import psycopg2 as pg

def connect():
    conn = None
    try:
        print('Connecting to the PostgreSQL database...')
        conn = pg.connect(
            database="ppi",
            user="postgres",
            password="password")
        cur = conn.cursor()
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
        db_version = cur.fetchone()
        print(db_version)
        cur.close()
    except(Exception, pg.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

if __name__ == '__main__':
    connect()
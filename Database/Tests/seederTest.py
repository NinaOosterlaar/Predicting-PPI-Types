import psycopg2 as pg


seeds = [
    "INSERT INTO go_term_category values (1, 'biological process');", # go_term_category 1
    "INSERT INTO go_term_category values (2, 'moluculair function');", # go_term_category 2 
    "INSERT INTO go_term values ('GO00001', 'GTPase activity', 1);", # go_term 1
    "INSERT INTO go_term values ('GO00002', 'GTPase activity', 2);" # go_term 2
    "INSERT INTO go_term_relation values (1, 'is_a');", # go_term_relation
    "INSERT INTO go_term_go_term values (1, 'GO00001', 'GO00002', 1);", # go_term_go_term
    "INSERT INTO organism values (1, 'Yeast');", # organism
    "INSERT INTO gene values ('YAL1234', 1, 'ADE2');", # gene 1
    "INSERT INTO gene values ('YAL5678', 1, 'CDC42');", # gene 2
    "INSERT INTO gene_go_term values (1, 'YAL1234', 'GO00001')", # gene_go_term
    "INSERT INTO ppi_type values (1, 'activation');", # ppi_type
    "INSERT INTO ppi values (1, 'YAL1234', 'YAL5678', 1);", # ppi
]
            

commands = (
    """
        SELECT gene.r_name, gt.term
        FROM gene, gene_go_term as ggt, go_term as gt
        WHERE gene.r_name = 'ADE2' AND gene.s_name = ggt.s_name AND ggt.go_id = gt.id;
    """,
)

            

delete = [
    "DELETE FROM gene WHERE s_name='YAL1234';", # go_term_go_term
    "DELETE FROM go_term_relation WHERE id=1;", # go_term_relation
    "DELETE FROM gene_go_term WHERE id=1", # gene_go_term
    "DELETE FROM ppi WHERE id=1 RETURNING *;", # ppi
    "DELETE FROM go_term WHERE id='GO00001';", # go_term 1
    "DELETE FROM go_term WHERE id='GO00002';" # go_term 2
    "DELETE FROM gene WHERE s_name='YAL1234';", # gene 1
    "DELETE FROM gene WHERE s_name='YAL5678';", # gene 2
    "DELETE FROM go_term_category WHERE id=1;", # go_term_category 1
    "DELETE FROM go_term_category WHERE id=2;", # go_term_category 2 
    "DELETE FROM organism WHERE id=1;", # organism
    "DELETE FROM ppi_type WHERE id=1;", # ppi_type  
]


def run(parameters, select = False):
    conn = None
    try:
        conn = pg.connect(
            database="ppi",
            user="postgres",
            password="password")
        cur = conn.cursor()
        for parameter in parameters:
            cur.execute(parameter)
        if select:
            print(cur.fetchall())
        conn.commit()
        cur.close()
    except (Exception, pg.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            
            
if __name__ == '__main__':
    run(seeds)
    run(commands, True)
    run(delete)

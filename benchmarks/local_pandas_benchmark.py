import datetime
import time

import psycopg

import pandas as pd
import pandas.io.sql as psql

_PATH = 'datasets/'

class Database:
    def __init__(self, host, dbname, schema, port, user, pwd):
        self.con = psycopg.connect(
            f'''host={host} 
            dbname={dbname} 
            user={user} 
            port={port} 
            password={pwd}'''
        )
        self.schema = schema

    def setBufferSize(self, size):
        self.con.replysize = size

    def get_table(self, name):
        t = pd.DataFrame(psql.read_sql_query('SELECT * FROM {0}.{1};'.format(self.schema, name), self.con))
        return t


def read_file(table):
    return pd.read_csv(''+_PATH+table+'.csv')


def my_query_01():
    """
    267.44
    (q_03) get the revenue for orders before 1995-03-15. order join with lineitem
    @param db:
    @return:
    """
    # local orders
    o = read_file('orders')
    l = my_db.get_table('lineitem')
    t = o.merge(l, left_on='o_orderkey', right_on='l_orderkey')[['l_orderkey', 'o_orderdate', 'o_shippriority']]
    t = t.groupby(('l_orderkey', 'o_orderdate', 'o_shippriority')).agg('count')
    print(t)
    return t

def my_query_01_r():
    """
    66.72
    (q_03) get the revenue for orders before 1995-03-15. order join with lineitem
    plan = (postgres -> (local, postgres))
    @param db:
    @return:
    """
    l = read_file('lineitem')
    o = my_db.get_table('orders')
    t = o.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t.groupby(('l_orderkey', 'o_orderdate', 'o_shippriority')).agg('count')
    return t

def my_query_02():
    """
    256.23
    (q_03) get the revenue for orders before 1995-03-15. order join with lineitem
    @param db:
    @return:
    """
    o = read_file('orders')
    l = my_db.get_table('lineitem')
    pivot_date = datetime.date(1998, 9, 2)
    l = l[l['l_shipdate'] < pivot_date]
    t = o.merge(l, left_on='o_orderkey', right_on='l_orderkey')[['l_orderkey', 'o_orderdate', 'o_shippriority']]
    return t

def q_01_v1():
    """291"""
    l = pd.read_remote_data('p1', 'lineitem')
    l = l.query(f'l_shipdate <= \'1998-9-2\'')
    l.sort_values(['l_returnflag', 'l_linestatus'])
    return l


def q_01_v2(db):
    l = db.lineitem
    l = l[l['l_shipdate'] <= datetime.date(1998, 9, 2)]
    l['disc_price'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l['charge'] = l['disc_price'] * (1 + l['l_tax'])
    l = l[['l_returnflag', 'l_linestatus', 'l_quantity', 'l_extendedprice', 'disc_price', 'charge', 'l_discount']]
    l = l.groupby(('l_returnflag', 'l_linestatus'), sort=False) \
        .agg({'l_quantity': ['sum', 'mean'], 'l_extendedprice': ['sum', 'mean'], 'disc_price': 'sum',
        'charge': 'sum', 'l_discount': ['mean', 'count']})
    l.columns = pd.Index([l.columns.levels[1][l.columns.labels[1][i]] + '_' + l.columns.levels[0][l.columns.labels[0][i]] for i in range(len(l.columns.labels[0]))])
    l.rename(columns={'sum_l_quantity': 'sum_qty', 'sum_l_extendedprice': 'sum_base_price',
        'mean_l_quantity': 'avg_qty', 'mean_l_extendedprice': 'avg_price',
        'mean_l_discount': 'avg_disc', 'count_l_discount': 'count_order'}, inplace=True)
    l.reset_index(inplace=True)
    l.sort_values(['l_returnflag', 'l_linestatus'], inplace=True)
    return l

def q_03_v1():
    o = read_file('orders')
    o = o.query('o_orderdate < \'1995-3-15\'')
    c = my_db.get_table('customer')

    l = my_db.get_table('lineitem')
    c = c.query('c_mktsegment == \'BUILDING\'')
    o = o[['o_orderdate', 'o_shippriority', 'o_orderkey', 'o_custkey']]
    l = l[l['l_shipdate'] > datetime.date(1995, 3, 15)]
    # l['revenue'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l = l[['l_orderkey', 'l_extendedprice']]

    t = c.merge(o, left_on='c_custkey', right_on='o_custkey')
    t = t.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t[['l_orderkey', 'l_extendedprice', 'o_orderdate', 'o_shippriority']]
    t = t.groupby(('l_orderkey', 'o_orderdate', 'o_shippriority')).agg('sum')

    return t

def measure_time(func, *args):
    start = time.time()
    rs = func(*args)
    end = time.time()
    print(rs)
    print('Function {} takes time {}'.format(func, end-start))


my_db = Database('localhost', 'sf01', 'sf01', 6000, 'sf01', 'sf01')

if __name__ == '__main__':
    measure_time(q_03_v1)
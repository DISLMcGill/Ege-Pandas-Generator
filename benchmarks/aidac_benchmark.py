import datetime
import marshal
import pickle
import time

import psycopg

import aidac
import aidac as pd
import pandas.io.sql as psql

_PATH = 'datasets/'


def connect(host, dbname, schema, port, user, pwd):
    aidac.add_data_source('postgres', host, user, pwd, dbname, 'p1', port)


def read_file(table):
    return pd.read_csv(_PATH+table+'.csv')

class My_UDF_Class:
    def my_func(self, a, b):
        return a+ b

def udf_func():
    import dill
    def my_udf(a, b):
        return a+b
    serilaized = marshal.dumps(my_udf.__code__)
    ds = aidac.manager.get_data_source('p1')
    print(serilaized)
    ds._execute('''INSERT INTO my_funcs VALUES ('my_udf', %s)''', serilaized)


def my_query_01():
    """
    140.88
    (q_03) get the revenue for orders before 1995-03-15. order join with lineitem
    plan = (postgres -> (local, postgres))
    @param db:
    @return:
    """
    o = read_file('orders')
    l = pd.read_remote_data('p1', 'lineitem')
    t = o.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t.groupby(('l_orderkey', 'o_orderdate', 'o_shippriority')).agg('count')
    return t

def my_query_01_r():
    """
    69.96 (40.60 + 4.41)
    (q_03) get the revenue for orders before 1995-03-15. order join with lineitem
    plan = (postgres -> (local, postgres))
    @param db:
    @return:
    """
    l = read_file('lineitem')
    o = pd.read_remote_data('p1', 'orders')
    t = o.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t.groupby(('l_orderkey', 'o_orderdate', 'o_shippriority')).agg('count')
    return t

def my_query_02():
    """
    209+38"""
    o = read_file('orders')
    l = pd.read_remote_data('p1', 'lineitem')
    l = l.query(f'l_shipdate < \'1998-9-2\'')
    t = o.merge(l, left_on='o_orderkey', right_on='l_orderkey')[['l_orderkey', 'o_orderdate', 'o_shippriority']]
    return t

def q_01_v1():
    l = pd.read_remote_data('p1', 'lineitem')
    l = l.query(f'l_shipdate <= \'1998-9-2\'')
    l.sort_values(['l_returnflag', 'l_linestatus'])
    return l

def q_01_v2():
    """209+38"""
    o = read_file('orders')
    l = pd.read_remote_data('p1', 'lineitem')
    l = l.query(f'l_shipdate < \'1998-9-2\'')
    t = o.merge(l, left_on='o_orderkey', right_on='l_orderkey')[['l_orderkey', 'o_orderdate', 'o_shippriority']]
    return t

def q_03_v1():
    c = pd.read_remote_data('p1', 'customer')
    o = pd.read_remote_data('p1', 'orders')
    l = pd.read_remote_data('p1', 'lineitem')
    c = c.query('c_mktsegment == \'BUILDING\'')['c_custkey']
    o = o.query('o_orderdate < \'1995-3-15\'')
    o = o[['o_orderdate', 'o_shippriority', 'o_orderkey', 'o_custkey']]
    l = l.query('l_shipdate > \'1995, 3, 15\'')
    # l['revenue'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l = l[['l_orderkey', 'l_extendedprice']]

    t = c.merge(o, left_on='c_custkey', right_on='o_custkey')
    t = t.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t[['l_orderkey', 'l_extendedprice', 'o_orderdate', 'o_shippriority']]
    t = t.groupby(('l_orderkey', 'o_orderdate', 'o_shippriority')).agg('sum', ['l_extendedprice'])
    # t.sort_values(['revenue', 'o_orderdate'], ascending=[False, True], inplace=True)
    # t.sort_values(['l_extendedprice', 'o_orderdate'], ascending=[False, True])
    print(t.genSQL)
    return t

def measure_time(func, *args):
    start = time.time()
    rs = func(*args).materialize()
    # rs = rs.materilazie()
    end = time.time()
    print(rs)
    print('Function {} takes time {}'.format(func, end-start))


if __name__ == '__main__':
    connect('localhost', 'sf01', 'sf01', 6000, 'sf01', 'sf01')
    measure_time(q_03_v1)
    # udf_func()
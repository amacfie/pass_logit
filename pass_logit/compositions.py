'''
uses code from

https://pypi.org/project/combalg-py/
https://pythonhosted.org/combalg-py/

License: MIT License (MIT)

Author: Sam Stump
'''

def all(n, k):
    '''
    A generator that returns all of the compositions of n into k parts.
    :param n: integer to compose
    :type n: int
    :param k: number of parts to compose
    :type k: int
    :return: a list of k-elements which sum to n

    Compositions are an expression of n as a sum k parts, including zero
    terms, and order is important.  For example, the compositions of 2
    into 2 parts:

    >>> compositions(2,2) = [2,0],[1,1],[0,2].

    NOTE: There are C(n+k-1,n) partitions of n into k parts.  See:
    `Stars and Bars <https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)>`_.
    '''
    t = n
    h = 0
    a = [0]*k
    a[0] = n
    yield tuple(a)
    while a[k-1] != n:
        if t != 1:
            h = 0
        t = a[h]
        a[h] = 0
        a[0] = t-1
        a[h+1] += 1
        h += 1
        yield tuple(a)

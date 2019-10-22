## Market Basket Analysis ##
## Association Rule Mining ##
## Finding Similar Items ##
# https://towardsdatascience.com/a-gentle-introduction-on-market-basket-analysis-association-rules-fa4b986a40ce
# https://discourse.snowplowanalytics.com/t/market-basket-analysis-identifying-products-and-content-that-go-well-together/1132
# dataset: http://archive.ics.uci.edu/ml/machine-learning-databases/00352/
# orders: https://www.kaggle.com/c/instacart-market-basket-analysis/data

# create rules of the items which have high support (most 
# frequently present in our dataset), because these will
# lead to large number of transactions

## support(a-->b) = fraction(a and b)/total_transactions
## confidence(a-->b) = support(a-->b)/support(a) = prob(b/a)
## lift(a-->b) = confidence(a-->b)/support(b)
## lift(a-->b) = support(a-->b)/support(b)*support(a)

# if lift is greater than 1, it suggests that presence of items on the
# left hand side has increased the likelihood of the items on the right
# hand side to be present in the transaction
# if the lift is less than 1, it suggests that presence of items on the
# left hand side has decreased the probability of the items on the right
# hand side to be present in the transaction
# if the lift is equal to 1 it means that the presence of items on the 
# LHS and RHS are independent

# when we perform market basket analysis, we look for the rules where 
# the lift is greater than 1 and rules also have a high support, and confidence


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

#read the retail transaction data
retail = pd.read_excel(r"Online Retail.xlsx")

retail = retail.dropna().reset_index(drop=True)
retail["Description"] = retail["Description"].apply(str.strip)

retail["DayOfWeek"] = retail.InvoiceDate.dt.day_name()
retail["Hour"] = retail.InvoiceDate.dt.hour

# what time of day does people buy products
retail["Hour"].hist()

# what day people buys
retail["DayOfWeek"].value_counts()

# how many items people buy
retail.groupby("InvoiceNo").Quantity.mean().hist(bins=100000)
retail.Quantity.max()
retail.Quantity.min()

def agg(x):
    return 1

retail = retail[retail.Country=="France"].reset_index(drop=True)

# get the baskets
pivot = pd.pivot_table(retail, index=['InvoiceNo'], columns=['Description'], values='Quantity', aggfunc=agg, fill_value=0)

## another way of getting the baskets
#basket = (df[df['Country'] =="France"]
#          .groupby(['InvoiceNo', 'Description'])['Quantity']
#          .sum().unstack().reset_index().fillna(0)
#          .set_index('InvoiceNo'))

frequent_item_sets = apriori(pivot, min_support=0.03, use_colnames=True)
rules = association_rules(frequent_item_sets, metric='lift', min_threshold=1)

rules.tail()


################# Another approach #####################
import numpy as np
from itertools import groupby, combinations
from collections import Counter
import sys

def get_size(obj):
    return sys.getsizeof(obj)/(1024*1024)

orders = pd.read_csv(r"order_products__prior.csv")
print("Memory: ", get_size(orders), "MB")
orders = orders.set_index('order_id')['product_id'].rename('item_id')
print("Total rows: ", len(orders))
print("Unique orders: ", orders.index.nunique())
print("Unique products: ", orders.nunique())

def freq(iterable):
    #returns frequency count of items and item pairs
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else:
        return pd.Series(Counter(iterable)).rename("freq")

def order_count(order_item):
    #returns count of unique orders
    return len(set(order_item.index))

def get_item_pairs(order_item):
    #returns generator that yields item pairs one at a time
    order_item = order_item.reset_index().as_matrix()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]
        for item_pair in combinations(item_list, 2):
            yield item_pair

def merge_item_name(rules, item_name):
    #returns name associated with item
    columns = ['itemA', 'itemB', 'freqAB', 'supportAB', 'freqA', 'supportA', 'freqB', 'supportB', 'confidenceAtoB', 'confidenceBtoA', 'lift']
    rules = rules.merge(item_name.rename(columns={'item_name':'itemA'}), left_on='item_A', right_on='item_id').\
        merge(item_name.rename(columns={'item_name':'itemB'}), left_on='item_B', right_on='item_id')
    return rules[columns]

def association_rules(order_item, min_support=0.01):

    print("Starting order_item: {:22d}".format(len(order_item)))


    # Calculate item frequency and support
    item_stats = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Filter from order_item items below min support 
    qualifying_items = item_stats[item_stats['support'] >= min_support].index
    order_item = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:5d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Filter from order_item orders with less than 2 items
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Recalculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(order_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))


    # Filter from item_pairs those below min support
    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    
    
    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)

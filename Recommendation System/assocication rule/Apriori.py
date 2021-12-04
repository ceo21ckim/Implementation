import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import time

imp = [
    ['맥주','오징어','치즈'],
    ['소주','맥주','라면'],
    ['맥주','오징어','사이다','콜라'],
    ['사이다','오징어','라면'],
    ['치즈','라면','계란']
]

te = TransactionEncoder().fit(imp)
te_ary = te.transform(imp)

print(te_ary)

# [[False False  True False False  True  True False]
#  [False  True  True False  True False False False]
#  [False False  True  True False  True False  True]
#  [False  True False  True False  True False False]
#  [ True  True False False False False  True False]]


implicit = pd.DataFrame(te_ary, columns = te.columns_ ,index = ['홍길동','고길동','이순신','강감찬','정약용'])
implicit

#         계란   라면    맥주  사이다   소주   오징어  치즈   콜라
# 홍길동  False  False   True  False  False   True   True  False
# 고길동  False   True   True  False   True  False  False  False
# 이순신  False  False   True   True  False   True  False   True
# 강감찬  False   True  False   True  False   True  False  False
# 정약용   True   True  False  False  False  False   True  False



freq_items = apriori(implicit, min_support = 0.4, use_colnames = True)
freq_items

freq_items = apriori(implicit, min_support = 0.2, use_colnames = True)
freq_items



from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

# association rule을 사용하면 각각에 대한 confidence lift 값을 확인할 수 있다. 
df = association_rules(freq_items, metric = 'confidence', min_threshold= 0.3)

df[:10]


df = fpgrowth(implicit, min_support= 0.2, use_colnames= True )
df
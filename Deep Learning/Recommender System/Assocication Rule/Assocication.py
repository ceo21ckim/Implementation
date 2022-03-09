from mlxtend.frequent_patterns import association_rules
import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

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


implicit = pd.DataFrame(te_ary, columns = te.columns_ ,index = ['홍길동','고길동','이순신','강감찬','정약용'])
implicit

freq_items = apriori(implicit, min_support = 0.2, use_colnames = True)
freq_items

ass = association_rules(freq_items, metric = 'confidence', min_threshold= 0.3)

ass
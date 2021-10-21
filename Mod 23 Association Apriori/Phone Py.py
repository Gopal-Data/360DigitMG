import pandas as pd
from mlxtend.frequent_patterns  import apriori, association_rules
import matplotlib.pyplot as plt
import os

#imported the data
phone = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 23\\myphonedata.csv")
phone=phone.drop(['V1','V2','V3'], axis=1) 

#Using the Apriori on Data
frequent_items = apriori(phone, min_support = .05 ,max_len = 3, use_colnames = True,verbose=1)

#Sorting the frequent items set based on support
frequent_items.sort_values('support',ascending= False, inplace = True)

#Using the Associate Rules
rules = association_rules(frequent_items,metric="lift",min_threshold = 1)
#18 Rules created with support .07 Maxlen 3

#sorting the data in Descending order with lift value
rules.sort_values('lift', ascending = False, inplace = True)
rules.to_csv('phone_Rules.csv')

#Removing the reducdencies in the rules
def to_list(i):
    return sorted(i)
ma_x = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_x = ma_x.apply(sorted)
return_rules = list(ma_x)
unique_rules = [list(m) for m in set(tuple(i) for i in return_rules)]

index_rules = []
for i in unique_rules:
    index_rules.append(return_rules.index(i))

#Getting the rules without any reducdancies
rules_no_redudancy = rules.iloc[index_rules, : ]

#Sorting them with respect to lift 
rules_no_redudancy.sort_values('lift', ascending = False) 
rules_no_redudancy.to_csv('Phonewithout.csv')
os.getcwd()

#Scatter plot to find the best dot the 
plt.scatter(x=rules_no_redudancy.support, y=rules_no_redudancy.confidence, c=rules_no_redudancy.lift , cmap = 'gray')
plt.colorbar()
plt.xlabel("support")
plt.ylabel("confidence")
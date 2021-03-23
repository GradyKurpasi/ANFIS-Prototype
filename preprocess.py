import numpy as np
import pandas as pd
from math import floor

master = pd.read_csv('master output.csv')
payments = pd.read_csv('payment output.csv')

#convert date columns to Pandas datetime datatypes (to assist with sorting)
master['dateplaced'] = pd.to_datetime(master['dateplaced'])
payments['DatePosted']= pd.to_datetime(payments['DatePosted'])

#sort both dataframes
master = master.sort_values(by=['customerid', 'accountid'])
payments = payments.sort_values(by=['customerid', 'accountid', 'DatePosted'])


output = pd.DataFrame()
record = 0

for index, row in master.iterrows():

    #get current account data    
    account_id = row['accountid']
    customer_id = row['customerid']

    
    #create blank payment history series
    pay_hist = pd.Series(dtype=object)
    for x in range(25):
        pay_hist = pay_hist.append(pd.Series({'Month'+str(x): 0}))

    #get payment rows
    account_payments = payments.loc[(payments['customerid'] == customer_id) & (payments['accountid']==account_id)]


    #if there are payments, fill in pay_hist 
    #only looking at first 24 months of payments
    for pay_index, pay_row in account_payments.iterrows():
        #get months between payment and account placement
        month = floor( (pay_row['DatePosted']-row['dateplaced']) / np.timedelta64(1, 'M') )
        if (month >= 0) and (month <= 24): #only look at first 24 months
            pay_hist['Month'+ str(month)] += pay_row['Collected'] #update pay history (will automatically subtract NSF)

    #calculate payment total and append to pay_hist
    pay_hist = pay_hist.append(pd.Series({'paytotal' : pay_hist.sum()}))

    #create output row and add to output data frame
    #row.append adds columns to series object row
    #output.append adds row to dataframe object output
    output = output.append(row.append(pay_hist), ignore_index=True)
    record += 1
    if record % 100 == 0:   print(record)
    tempfile = 'out' + str(record) + '.csv'
    if record % 100000 == 0: output.to_csv(tempfile, index=True)
    #DEBUG
    #print(output)

#write to file and finish
output.to_csv('output.csv', index=True)
print('DONE')
        




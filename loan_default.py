#import numpy as np
import pandas as pd
#from datetime import date, datetime
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import pickle

def calculate_age(dob):
    age = []
    for i in range(len(dob)):
        born = pd.to_datetime('today').year-pd.to_datetime(dob[i]).year
        age.append(born)
    return age


df = pd.read_csv(r"C:\Users\703247130\Documents\train_aox2Jxw\train.csv")
X = df.drop('loan_default',axis=1)
Y = df['loan_default']

corr = df.corr()

X['Employment.Type'].fillna('Self employed', inplace=True)
X['Employment.Type'] = le.fit_transform(X['Employment.Type'])
X['PERFORM_CNS.SCORE.DESCRIPTION'] = pd.get_dummies(X, columns=['PERFORM_CNS.SCORE.DESCRIPTION'])
X['AVERAGE.ACCT.AGE'] = X['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0])*12 + int(x[-4]))
X['CREDIT.HISTORY.LENGTH'] = X['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0])*12 + int(x[-4]))
X['Total.Emi'] = X['PRIMARY.INSTAL.AMT'] + X['SEC.INSTAL.AMT']
X['Age'] = calculate_age(X['Date.of.Birth'])
X.drop(['supplier_id','manufacturer_id','Current_pincode_ID','Date.of.Birth','DisbursalDate','State_ID','Employee_code_ID','UniqueID'], axis=1, inplace=True)
X.drop(['MobileNo_Avl_Flag'], axis=1, inplace=True)
X['disbursed_amount'] = (X['disbursed_amount']-X['disbursed_amount'].mean())/df['disbursed_amount'].std()
X['asset_cost'] = (X['asset_cost']-X['asset_cost'].mean())/df['asset_cost'].std()
X['ltv'] = (X['ltv']-X['ltv'].mean())/df['ltv'].std()
#X['Total.No.of.Acc'] = X['PRI.NO.OF.ACCTS'] + X['SEC.NO.OF.ACCTS']
X['Total.Active.Loans'] = X['PRI.ACTIVE.ACCTS'] + X['SEC.ACTIVE.ACCTS']
#X['Total.Overdue.Acc'] = X['PRI.OVERDUE.ACCTS'] + X['SEC.OVERDUE.ACCTS']
X['Total.Outstanding.Amt'] = X['PRI.CURRENT.BALANCE'] + X['SEC.CURRENT.BALANCE']
#X['Total.Sanctioned.Amt'] = X['PRI.SANCTIONED.AMOUNT'] + X['SEC.SANCTIONED.AMOUNT']
#X['Total.Disbursed.Amt'] = X['PRI.DISBURSED.AMOUNT'] + X['SEC.DISBURSED.AMOUNT']

#X.drop(['PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS','PRI.ACTIVE.ACCTS','SEC.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','SEC.OVERDUE.ACCTS','PRI.CURRENT.BALANCE','SEC.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT','SEC.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT','SEC.DISBURSED.AMOUNT'], axis=1, inplace=True)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

classifiers=[
    (DecisionTreeClassifier(criterion="entropy"),'DecisionTreeClassifier'),
    (RandomForestClassifier(),"RandomForestClassifier"),
    (BaggingClassifier(n_estimators=20),'BaggingClassifier'),
    (AdaBoostClassifier(DecisionTreeClassifier(criterion="entropy"),n_estimators=10),'AdaBoostClassifier'),
    (GaussianNB(),'GaussianNB')
]

final_dict = {}

def fscore(tp,fp,fn):
    precision = round(tp/(tp+fp),3)
    recall = round(tp/(tp+fn),3)
    fscore= round(2 * ((precision*recall)/(precision+recall)),3)
    print("Precision = {}, Recall = {}, F-1 Score = {}".format(precision,recall,fscore))
    

for model, name in classifiers:
    clf = model
    clf.fit(X_train, Y_train)
    print(clf.score(X_train, Y_train))
    fn = "model_"+name+"_1"+".sav"
    pickle.dump(clf, open("AV Loan_Default/"+fn, 'wb'))
    predict = clf.predict(X_test)
    Accuracy = round(accuracy_score(Y_test, predict)*100,3)
    z = confusion_matrix(Y_test, predict)
    final_dict[name] = Accuracy
    print(name)
    print ('Accuracy: {}'.format(Accuracy))
    fscore(z[0][0],z[0][1],z[1][0])
    print("Confusion Matrix")
    print(z)
    print("--------------<<>>------------------")
    
filename = ['AV Loan_Default/model_LogisticRegression.sav','AV Loan_Default/model_KNeighborsClassifier.sav','AV Loan_Default/model_DecisionTreeClassifier.sav','AV Loan_Default/model_GaussianNB.sav']

for fn in filename:
    print(fn)
    clf = pickle.load(open(fn, 'rb'))
    predict = clf.predict(X_test)
    Accuracy = round(accuracy_score(Y_test, predict)*100,3)
    z = confusion_matrix(Y_test, predict)
    print ('Accuracy: {}'.format(Accuracy))
    fscore(z[0][0],z[0][1],z[1][0])
    print("Confusion Matrix")
    print(z)
    print("--------------<<>>------------------")
    
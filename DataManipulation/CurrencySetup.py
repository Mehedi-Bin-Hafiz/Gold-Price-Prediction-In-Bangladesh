import pandas as pd

file = open('OutputedDataset.csv','a')


Currency = pd.read_csv('../Database/Currency.csv')
GoldPrice= pd.read_excel('../Database/GoldPrice.xlsx')


CYear=Currency.iloc[:, 3:4].values
CDate=Currency.iloc[:,1:2].values
CMonth=Currency.iloc[:,2:3].values
CValue=Currency.iloc[:,4:5].values

print(len(CDate))


Gyear = GoldPrice.iloc[:, :1].values
GDate=GoldPrice.iloc[:, 1:2].values
GMonth=GoldPrice.iloc[:, 2:3].values
print(len(GDate))

Glis=list()
Clis=list()

Matching=list()
for i in range(0,len(Gyear)):
    Glis.append([Gyear.item(i)+2000,GMonth.item(i),GDate.item(i)])
    Clis.append([CYear.item(i),CMonth.item(i),CDate.item(i)])

print(Glis)

for i in range(0,len(Glis)):
    if Clis[i] in Glis:
        Glis[Glis.index(Clis[i])].insert(3,CValue.item(i))
    else:
       pass
for i in Glis:
    if len(i) == 4:
        file.write("{2},{1},{0},{3}\n".format(i[0],i[1],i[2],i[3]))
    else:
        file.write("{2},{1},{0},{3}\n".format(i[0],i[1],i[2],0))

file.close()












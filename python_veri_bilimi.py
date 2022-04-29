import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import * 
from PyQt5.QtGui import QDoubleValidator
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 


class FloatDelegate(QItemDelegate):
    def __init__(self, parent=None):
        super().__init__()

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QDoubleValidator())
        return editor

class TableWidget(QTableWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.setStyleSheet('font-size: 15px;')
        # tablo boyutunu ayarlama
        nRows, nColumns = self.df.shape
        self.setColumnCount(nColumns)
        self.setRowCount(nRows)
        self.setHorizontalHeaderLabels(("Tarih","Gün","Gelir Açıklama","Gider Açıklama","Gelir Tutar","Gider Tutar"))
        self.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.setItemDelegateForColumn(1, FloatDelegate())

        # veri ekleme
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                self.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))

        self.cellChanged[int, int].connect(self.guncellemeDF)   

    def guncellemeDF(self, row, column):
        text = self.item(row, column).text()
        self.df.iloc[row, column] = text

class DFEditor(QWidget):
    

    veri =np.array([["01.01.2020",1,"Ürün Satışı","Fatura Bedeli",145,60],["02.01.2020",2,"Faiz Geliri","Personel Gideri",250,50],["03.01.2020",3,"Alacak Tahsilat","-",150,0],["04.01.2020",4,"Nakit Satış","Fatura Bedeli",400,160], ["05.01.2020",5,"-","Hammadde Alımı",0,300],
                    ["06.01.2020",6,"Çek Tahsilatı","Fatura Bedeli",500,300],["07.01.2020",7,"Ürün Satışı","Personel Gideri",140,65],["08.01.2020",8,"Alacak Tahsilatı","Fatura Bedeli",245,90],["09.01.2020",9,"Ürün Satışı","-",400,0], 
                    ["10.01.2020",10,"Çek Tahsilatı","Personel Gideri",100,30],["11.01.2020",11,"Çek Tahsilatı","Personel Gideri",100,30],["12.01.2020",12,"Ürün Satışı","-",400,0],["13.01.2020",13,"Alacak Tahsilatı","Fatura Bedeli",245,90],["14.01.2020",14,"Alacak Tahsilat","-",150,0],
                    ["15.01.2020",15,"Ürün Satışı","Fatura Bedeli",145,60],["16.01.2020",16,"Alacak Tahsilat","-",200,0],["17.01.2020",17,"Faiz Geliri","Personel Gideri",250,50],["18.01.2020",18,"Nakit Satış","Fatura Bedeli",400,160],
                    ["19.01.2020",19,"Ürün Satışı","Fatura Bedeli",125,50],["20.01.2020",20,"Faiz Geliri","Personel Gideri",250,50],["21.01.2020",21,"Faiz Geliri","Personel Gideri",250,50],["22.01.2020",22,"-","Hammadde Alımı",0,300],
                    ["23.01.2020",23,"Ürün Satışı","-",275,0],["24.01.2020",24,"Çek Tahsilatı","Fatura Bedeli",530,370], ["25.01.2020",25,"Alacak Tahsilatı","Fatura Bedeli",245,90],["26.01.2020",26,"Faiz Geliri","Personel Gideri",250,50],["27.01.2020",27,"Alacak Tahsilat","-",150,0],
                    ["28.01.2020",28,"Çek Tahsilatı","Personel Gideri",100,30],["29.01.2020",29,"Çek Tahsilatı","Personel Gideri",245,75],["30.01.2020",30,"Ürün Satışı","-",400,0]])
    df = pd.DataFrame(data=veri, columns =["Tarih","Gün","Gelir Açıklama","Gider Açıklama","Gelir Tutar","Gider Tutar"]) 
    
    

    def __init__(self):
        super().__init__()
        self.setGeometry(1400,900,1200,900)
        self.move(50,70)
        self.setWindowTitle("Pimapen(PCV) Gelir-Gider Defteri".upper())

        mainLayout = QVBoxLayout()

        self.table = TableWidget(DFEditor.df)
        mainLayout.addWidget(self.table)

        button_print = QPushButton('DF Terminale Yazdır')
        button_print.setStyleSheet('font-size: 30px')
        button_print.clicked.connect(self.DF_degerlerini_yazdir)
        mainLayout.addWidget(button_print)

        button_export = QPushButton('CSV Dosyasına Aktar')
        button_export.setStyleSheet('font-size: 30px')
        button_export.clicked.connect(self.CSV_dosyasina_aktar)
        mainLayout.addWidget(button_export)     

        self.setLayout(mainLayout)
        
    def DF_degerlerini_yazdir(self):
        print(self.table.df)

    def CSV_dosyasina_aktar(self):
        self.table.df.to_csv('verileri kaydet.csv', index=False)
        print('CSV dosyası dışa aktarıldı')


uyg = QApplication(sys.argv)
demo = DFEditor()
demo.show()
sys.exit(uyg.exec_())





dataset=pd.read_csv('C:\csv dosyası\dataset.csv')
print(dataset.head()) 
print(dataset.tail())
print(dataset.describe()) 
dataset.plot(x='Gün',y='Gelir Tutar',style='o') 
plt.title("Günlere Göre Gelirler") 
plt.xlabel("Günler") 
plt.ylabel("Gelir Tutarı") 
plt.show() 

X=dataset[['Gün','Gider Tutar']] 
y=dataset[['Gelir Tutar']] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
regressor = LinearRegression() 
regressor.fit(X_train, y_train)#bu noktaları kullanarak düz bir çizgi oluşturmak fit etmek
print(regressor.coef_) 
print(regressor.intercept_)#kesişmeler
y_pred=regressor.predict(X_test) 
print("Gerçek Değerler:",y_test) 
print("Tahmini Değerler:",y_pred) 
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred)) 
print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_pred)) 
print("Root Mean Squared Error:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))







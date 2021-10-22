# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:38:51 2020

@author: soori
"""
########################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
#from jupyterthemes import jtplot
import seaborn as sns
#import datetime
import calendar
import math
from datetime import datetime
from flask import Flask, render_template, request
#from werkzeug import secure_filename
from warnings import filterwarnings
filterwarnings('ignore')

app = Flask(__name__)
#jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 


########################################################################################################

global df1 
df1=pd.DataFrame()

@app.route('/')
def upload_file():
   return render_template('mp1.html')

@app.route('/temp', methods = ['GET', 'POST'])
def tmp():
    global df1
    if request.method == 'POST':
        f=request.files['file']
        #f.save(f.filename)
        #print("F=",f)
        #print("type of F=",type(f))
        df1=pd.read_excel(f)
        if df1.columns[0]=="InvoiceNo" and df1.columns[1]=="StockCode" and df1.columns[2]=="Description" and df1.columns[3]=="Quantity" and df1.columns[4]=="InvoiceDate" and df1.columns[5]=="UnitPrice" and df1.columns[6]=="CustomerID" and df1.columns[7]=="Country":
            return render_template("mp3.html")
        else:
            return "The column names does not matches with the mentioned column names...Kindly use same column names as mentioned in home page"
        #print("F=",f.type())
        
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
   global df1
   if request.method == 'POST':
# =============================================================================
#       f = request.files['file']
#       #cls=int(request.form['clusters'])
#       df1=pd.read_excel(f)
# =============================================================================
      
      df1=df1.dropna()
      df1.drop_duplicates(keep=False,inplace=True)  #dropping duplicate records
      df1.drop(df1[df1['Quantity'] < 1].index,inplace=True)     #eliminating negative quantities
      
      df1.reset_index(inplace=True,drop=True)
      
      descunique=df1['Description'].unique()
      descunique=list(descunique)
      
      custidunique=df1['CustomerID'].unique()
      custidunique=list(custidunique)
      
      tot=[]
      for i in range(len(descunique)):
          tmpdf1=df1[df1['Description']==descunique[i]]
          tmp=[]
          for i in range(len(tmpdf1)):
              tmp.append(tmpdf1['CustomerID'].iloc[i])
          t1=set(tmp)
          tmp=list(t1)
          tot.append(tmp)
          
      totprice=[]
      for i in range(len(df1)):
          tmp4=df1['Quantity'][i]*df1['UnitPrice'][i]
          tmp4=round(tmp4,2)
          totprice.append(tmp4)
      
      df1['Totprice']=totprice
      
      finaltopcust=[]
      totcust=[]
      for i in range(len(descunique)):
          tmp2=tot[i]
          tmpdf1=df1[df1['Description']==descunique[i]]
          cust=[]
          for j in range(len(tmp2)):
              t1=[]
              t2=[]
              
              t1=tmpdf1.loc[tmpdf1['CustomerID']==tmp2[j],"Quantity"]
              tot_qnt=0
              for k in t1:
                  tot_qnt=tot_qnt+k
                  
              t2=tmpdf1.loc[tmpdf1['CustomerID']==tmp2[j],"Totprice"]
              tot_price=0
              for k in t2:
                  tot_price=tot_price+k
              
              cust.append([descunique[i],tmp2[j],tot_qnt,round(tot_price,2)])
              cust.sort(key = lambda x: x[2])
          totcust.append(cust)
          op=[]
          if len(cust)>2:
              op.append(cust[-1])
              op.append(cust[-2])
              op.append(cust[-3])
          elif len(cust)==2:
              op.append(cust[-1])
              op.append(cust[-2])
          elif len(cust)==1:
              op.append(cust[-1])
          
          finaltopcust.append(op)

# =============================================================================
#       qnt1=[]
#       unt1=[]
#       for i in range(len(totcust[0])):
#           qnt1.append(totcust[0][i][1])
#           unt1.append(totcust[0][i][2])
# =============================================================================
      
      alldates=[]
      for i in range(len(df1)):
          tmp4=df1['InvoiceDate'][i].date()
          alldates.append(tmp4)
        
      datepricedf=pd.DataFrame()
      datepricedf['Dates']=alldates
      datepricedf['Totprice']=totprice
      datepricedf=datepricedf.sort_values('Totprice')
      datepricedf.reset_index(inplace = True, drop = True) 
      
      dailydates=set(alldates)
      dailydates=list(dailydates)
      dailyprice=[]
      #dailydateprice=[]
      
      tdf=pd.DataFrame()
      for i in range(len(dailydates)):
          tqnt=0
          tdf.drop(tdf.index, inplace=True)
          tdf=datepricedf[datepricedf['Dates']==dailydates[i]].reset_index()
          for j in range(len(tdf)):
              tqnt=tqnt+tdf['Totprice'][j]
          tqnt=round(tqnt,2)
          dailyprice.append(tqnt)
          #dailydateprice.append([dailydates[i],tqnt])
      
      plt.figure(figsize=(20,20))
      plt.xlabel("Dates",size=20)
      plt.ylabel("Sales amount",size=20)
      plt.title("Daily sales",size=20)
      plt.bar(dailydates, dailyprice,width=0.5) 
      
      pth="static/dailysales.png"
      plt.savefig(pth)
      plt.clf()
      
      #print("Dailysales done")
      
      tdf=pd.DataFrame()
      tdf.drop(tdf.index,inplace=True)
      alltime=[]
      for i in range(len(df1)):
          tmp=df1['InvoiceDate'][i].time().hour
          tmp2=df1['Totprice'][i]
          alltime.append([tmp,tmp2])
          
      tdf2=pd.DataFrame()
      tdf2 = pd.DataFrame(alltime, columns = ['Time', 'Finalprice'])
      
      finalhrsprice=[]
      tdf=pd.DataFrame()
      for i in range(1,25):
          tdf.drop(tdf.index,inplace=True)
          tdf=tdf2[tdf2['Time']==i]
          tdf.reset_index(inplace=True,drop=True)
          price=0
          for j in range(len(tdf)):
              price+=tdf2['Finalprice'][j]
          finalhrsprice.append([i,round(price,2)])
          
      finalhrsprice2=[]
      for i in range(24):
          tmp=finalhrsprice[i][1]
          if tmp!=0:
              finalhrsprice2.append(finalhrsprice[i])
              
      tdf=pd.DataFrame()
      tdf.drop(tdf.index,inplace=True)
      tdf=pd.DataFrame(finalhrsprice2,columns=['Hrs','Totsales'])
      
      #plt.figure(figsize=(20,10))
      plt.xlabel("Hours",size=20)
      plt.ylabel("Sales amount",size=20)
      plt.title("Hourly total sales",size=20)
      a1=plt.bar(tdf['Hrs'], tdf['Totsales'],width=0.5) 
      
      for rect in a1:
          height = rect.get_height()
          plt.text(rect.get_x() + rect.get_width()/2, 1.01*height,
                   height,
            ha='center', va='bottom')
          
      pth="static/hrssales.png"
      plt.savefig(pth)
      plt.clf()
      
      #dailydateprice.sort(key = lambda x: x[0])
      
      ddpdf=pd.DataFrame()
      ddpdf['Dates']=dailydates
      ddpdf['Totprice']=dailyprice
      #ddpdf.reset_index(inplace=True)
      ddpdf.sort_values('Dates',inplace=True)
      ddpdf.reset_index(inplace=True,drop=True)
      
      days=[]
      for i in range(len(ddpdf)):
          tmp=ddpdf['Dates'][i]
          day=datetime.strptime(str(tmp), '%Y-%m-%d').weekday() 
          day=calendar.day_name[day]
          days.append(day)
      
      ddpdf['Days']=days
     
      arrdays=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
      arrprice=[]
      tmprdf=pd.DataFrame()
      for i in range(7):
          tmprdf.drop(tmprdf.index,inplace=True)
          tmprdf=ddpdf[ddpdf['Days']==arrdays[i]].reset_index()
          ttp=0
          #print(tmprdf['Totprice'][0])
          for j in range(len(tmprdf)):
              ttp+=tmprdf['Totprice'][0]
          arrprice.append(round(ttp,2))
      
      #plt.figure(figsize=(20,10))
      a1=plt.bar(arrdays,arrprice,width=0.5)
      plt.xlabel("Days",size=25)
      plt.ylabel("Total sales price",size=25)
      plt.title("Total sales by days",size=25)
      
      for rect in a1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height, height, ha='center', va='bottom',size=20)
      
      pth="static/total-sales-by-days.png"
      plt.savefig(pth)
      plt.clf()
      
      #print("Total sales by days done")
      
      allmthyr=[]
      srtdailydates=sorted(dailydates)
      for i in range(len(sorted(dailydates))):
          mth=srtdailydates[i].month
          yr=srtdailydates[i].year
          mthyr=str(yr)+"/"+str(mth)
          allmthyr.append(mthyr)
      allmthyrset=list(set(allmthyr))
      allmthyrset.sort(key = lambda date: datetime.strptime(date, '%Y/%m'))
      
      ddpdf['Dt']=allmthyr
      
      arrtotmthprice=[]
      tdf=pd.DataFrame()
      for i in range(len(allmthyrset)):
          tdf.drop(tdf.index,inplace=True)
          tdf=ddpdf[ddpdf['Dt']==allmthyrset[i]].reset_index()
          totmthprice=0
          for j in range(len(tdf)):
              totmthprice+=tdf['Totprice'][j]
          arrtotmthprice.append(round(totmthprice,2))

      #plt.figure(figsize=(20,10))
      a1=plt.bar(allmthyrset,arrtotmthprice,width=0.5)
      plt.xlabel("Months",size=25)
      plt.ylabel("Amount of sales",size=25)
      plt.title("Monthly sales",size=25)
      
      for rect in a1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height, height, ha='center', va='bottom')
          
      pth="static/monthly_sales.png"
      plt.savefig(pth)
      plt.clf()
      
      #print("Monthly sales done")

# =============================================================================
#       tdf=pd.DataFrame()
#       allqntsales=[]
#       
#       for i in range(len(descunique)):
#           tdf.drop(tdf.index,inplace=True)
#           tdf=df1[df1['Description']==descunique[i]]
#           tdf.reset_index(inplace=True,drop=True)
#           qntsales=0
#           for j in range(len(tdf)):
#               qntsales+=tdf['Quantity'][j]
#           allqntsales.append([descunique[i],qntsales])
# 
#       allqntsales.sort(key=lambda x:x[1],reverse=True)
#       
#       allqntsales_top10=allqntsales[:10]
# =============================================================================

      tdf=df1.groupby('Description')['Quantity'].sum()
      tdf=tdf.reset_index()
      tdf.sort_values('Quantity',inplace=True,ascending=False)
      tdf.reset_index(inplace=True,drop=True)
      tdf=tdf.head(10)
      
      allqntsales_top10=[]
      for i in range(len(tdf)):
          tmp1=tdf['Description'][i]
          tmp2=tdf['Quantity'][i]
          allqntsales_top10.append([tmp1,round(tmp2,2)])
      
# =============================================================================
#       tdf=pd.DataFrame()
#       alltoppricesales=[]
#       
#       for i in range(len(descunique)):
#           tdf.drop(tdf.index,inplace=True)
#           tdf=df1[df1['Description']==descunique[i]]
#           tdf.reset_index(inplace=True,drop=True)
#           pricesales=0
#           for j in range(len(tdf)):
#               pricesales+=tdf['Totprice'][j]
#           alltoppricesales.append([descunique[i],round(pricesales,2)])
# 
#       alltoppricesales.sort(key=lambda x:x[1],reverse=True)
#       
#       alltoppricesales_top10=alltoppricesales[:10]
# =============================================================================

      tdf=df1.groupby('Description')['Totprice'].sum()
      tdf=tdf.reset_index()
      tdf.sort_values('Totprice',inplace=True,ascending=False)
      tdf.reset_index(inplace=True,drop=True)
      tdf=tdf.head(10)
      
      alltoppricesales_top10=[]
      for i in range(len(tdf)):
          tmp1=tdf['Description'][i]
          tmp2=tdf['Totprice'][i]
          alltoppricesales_top10.append([tmp1,round(tmp2,2)])
      
# =============================================================================
#       inv=df1['InvoiceNo']
#       
#       inv=list(inv)
#       invset=list(set(inv))
# 
#       allinvprice=[]
#       tdf=pd.DataFrame()
#       for i in range(len(invset)):
#           tdf.drop(tdf.index,inplace=True)
#           tdf=df1[df1['InvoiceNo']==invset[i]]
#           tdf.reset_index(inplace=True,drop=True)
#           invprice=0
#           for j in range(len(tdf)):
#               invprice+=tdf['Totprice'][j]
#           allinvprice.append([invset[i],tdf['CustomerID'][0],round(invprice,2)])
# 
#       allinvprice.sort(key=lambda x:x[2],reverse=True)
#       
#       allinvprice_top10=allinvprice[:10]
# =============================================================================

      tdf=df1.groupby('InvoiceNo')['Totprice'].sum()
      tdf=tdf.reset_index()
      tdf.sort_values('Totprice',inplace=True,ascending=False)
      tdf.reset_index(inplace=True,drop=True)
      tdf=tdf.head(10)
      
      allinvprice_top10=[]
      for i in range(len(tdf)):
          tmp1=tdf['InvoiceNo'][i]
          tmp2=df1.loc[df1['InvoiceNo']==tmp1,'CustomerID'].iloc[0]
          tmp3=tdf['Totprice'][i]
          allinvprice_top10.append([tmp1,tmp2,round(tmp3,2)])
      
# =============================================================================
#       topcust=[]
#       tdf=pd.DataFrame()
#       for i in range(len(custidunique)):
#           tdf.drop(tdf.index,inplace=True)
#           tdf=df1[df1['CustomerID']==custidunique[i]]
#           tdf.reset_index(inplace=True,drop=True)
#           finalprice=0
#           for j in range(len(tdf)):
#               finalprice+=tdf['Totprice'][j]
#           topcust.append([custidunique[i],round(finalprice,2)])
#       
#       topcust.sort(key=lambda x:x[1],reverse=True)
#       topcust_top10=topcust[:10]
# =============================================================================

      tdf=df1.groupby('CustomerID')['Totprice'].sum()
      tdf=tdf.reset_index()
      tdf.sort_values('Totprice',inplace=True,ascending=False)
      tdf.reset_index(inplace=True,drop=True)
      tdf=tdf.head(10)
      
      topcust_top10=[]
      for i in range(len(tdf)):
          tmp1=tdf['CustomerID'][i]
          tmp2=tdf['Totprice'][i]
          topcust_top10.append([tmp1,round(tmp2,2)])
      
# =============================================================================
#       cntryunique=list(df1['Country'].unique())
#       
#       cntryunique.sort()
#       
#       cntrywisesales=[]
#       cntsalesunique=[]
#       tdf1=pd.DataFrame()
#       for i in range(len(cntryunique)):
#           tdf1.drop(tdf1.index,inplace=True)
#           tdf1=df1[df1['Country']==cntryunique[i]]
#           tdf1.reset_index(inplace=True,drop=True)
#           tcntsales=0
#           for j in range(len(tdf1)):
#               tcntsales+=tdf1['Totprice'][j]
#           cntrywisesales.append([cntryunique[i],round(tcntsales,2)])
#           cntsalesunique.append(tcntsales)
#         
#       cntrywisesales.sort(key=lambda x:x[0])
# =============================================================================

      tdf=df1.groupby('Country')['Totprice'].sum()
      tdf=tdf.reset_index()
      tdf.reset_index(inplace=True,drop=True)
      tdf.Totprice = tdf.Totprice.round(2)
      
      #plt.figure(figsize=(20,20))
      a1=plt.barh(tdf['Country'],tdf['Totprice'], color='crimson')
      plt.title("Country wise overall sales price",size=20)
      plt.gca().invert_yaxis()
      
      for rect in a1:
          plt.text(rect.get_width()*1.02,rect.get_y()+0.5,str(round(rect.get_width(),2)),size=18)
    
      plt.margins(0.02)
      plt.grid(b=True, color='silver', linestyle='-.', linewidth=0.5, alpha=0.75)
        
      plt.rcParams['axes.facecolor'] = 'grey'
      
      pth="static/country_overall_sales.png"
      plt.savefig(pth)
      plt.clf()
      
      #print("Country wise overall sales done")
      finaltopcust.sort(key=lambda x:x[0][0])
      #print(topcust_top10)
      
      retail=df1
      retail['Amount'] = retail['Quantity']*retail['UnitPrice']
      rfm_m = retail.groupby('CustomerID')['Amount'].sum()
      rfm_m = rfm_m.reset_index()
      
      rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
      rfm_f = rfm_f.reset_index()
      rfm_f.columns = ['CustomerID', 'Frequency']
      
      rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
      
      retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')
      
      max_date = max(retail['InvoiceDate'])
      
      retail['Diff'] = max_date - retail['InvoiceDate']
      
      rfm_p = retail.groupby('CustomerID')['Diff'].min()
      rfm_p = rfm_p.reset_index()
      
      rfm_p['Diff'] = rfm_p['Diff'].dt.days
      
      rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
      rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
      
      # Removing (statistical) outliers for Amount
      Q1 = rfm.Amount.quantile(0.05)
      Q3 = rfm.Amount.quantile(0.95)
      IQR = Q3 - Q1
      rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]
      
      # Removing (statistical) outliers for Recency
      Q1 = rfm.Recency.quantile(0.05)
      Q3 = rfm.Recency.quantile(0.95)
      IQR = Q3 - Q1
      rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]
      
      # Removing (statistical) outliers for Frequency
      Q1 = rfm.Frequency.quantile(0.05)
      Q3 = rfm.Frequency.quantile(0.95)
      IQR = Q3 - Q1
      rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]
      
      # Rescaling the attributes

      rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
      
      # Instantiate
      scaler = StandardScaler()
      
      # fit_transform
      rfm_df_scaled = scaler.fit_transform(rfm_df)
      
      rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
      rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
      
      # Elbow-curve/SSD

      ssd = []
      range_n_clusters = [1,2, 3, 4, 5, 6, 7, 8]
      for num_clusters in range_n_clusters:
          kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
          kmeans.fit(rfm_df_scaled)
          
          ssd.append(kmeans.inertia_)
          
      def calc_dist(x1,y1,a,b,c):
        d=abs((a*x1 + b*y1 + c))/math.sqrt(a*a + b*b)
        return d
    
      a=ssd[0]-ssd[-1]
      b=range_n_clusters[-1] - range_n_clusters[0]
      c1=range_n_clusters[0] * ssd[-1]
      c2=range_n_clusters[-1] * ssd[0]
      c=c1-c2
      
      dist_of_points_from_line=[]
      for i in range(8):
          dist_of_points_from_line.append( calc_dist(range_n_clusters[i], ssd[i], a, b, c) )
          
      optval=dist_of_points_from_line.index(max(dist_of_points_from_line))+1
      #print(dist_of_points_from_line)
      #print(optval)
      
      # Final model with k=optval
      kmeans = KMeans(n_clusters=optval, max_iter=50)
      kmeans.fit(rfm_df_scaled)
      
      # assign the label
      rfm['Cluster_Id'] = kmeans.labels_
      
      # Box plot to visualize Cluster Id vs amount
      fig, ax = plt.subplots()
      fig.patch.set_facecolor('xkcd:mint green')
      sns.boxplot(x='Cluster_Id', y='Amount', data=rfm,ax=ax)
      
      pth="static/cid-amnt.png"
      plt.savefig(pth)
      fig.clear(True)
      #print("AAA")
      
      # Box plot to visualize Cluster Id vs Frequency
      fig, ax = plt.subplots()
      fig.patch.set_facecolor('xkcd:mint green')
      sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm,ax=ax)
      
      pth="static/cid-freq.png"
      plt.savefig(pth)
      #plt.clf()
      #print("BBB")
      
      # Box plot to visualize Cluster Id vs Recency
      fig, ax = plt.subplots()
      fig.patch.set_facecolor('xkcd:mint green')
      sns.boxplot(x='Cluster_Id', y='Recency', data=rfm,ax=ax)
      
      pth="static/cid-rec.png"
      plt.savefig(pth)
      #plt.clf()
      #print("CCC")
      
      cluster=rfm['Cluster_Id'].tolist()
      customers=rfm['CustomerID'].tolist()
      
      return render_template("mp2.html",descunique=sorted(descunique),
                                        finaltopcust=finaltopcust,
                                        allqntsales_top10=allqntsales_top10,
                                        alltoppricesales_top10=alltoppricesales_top10,
                                        allinvprice_top10=allinvprice_top10,
                                        topcust_top10=topcust_top10,
                                        cluster=cluster,
                                        customers=customers)

   return "Some error occurred!!! Kindly try again..."
      
#app.jinja_env.auto_reload = True
#app.config['TEMPLATES_AUTO_RELOAD'] = True
app.run(debug=True)
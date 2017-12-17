import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss
import seaborn as sns
from scipy.stats import mstats
import os
import statsmodels.api as sm
from sklearn.decomposition import PCA

def createImagesFolder(directory):

   if not os.path.exists(directory):
       os.makedirs(directory)


def applyPCA(df, components):
   pca = PCA(n_components=components)
   pca.fit(df[:-1])
   PCA(copy=True, iterated_power='auto', n_components=components, random_state=None,
   svd_solver='auto', tol=0.0, whiten=False)
   print "PCA levels :",(pca.explained_variance_ratio_) 



def multipleRegression(df):
  text_file = open("figures/RegressionMultipleValues.txt", "w")
  # regression for every pair of columns
  for i in range (len(df.columns.tolist())):
     for j in range(i+1, len(df.columns.tolist())-1): 
        X =  df[[df.columns[i],df.columns[j]]]
        y = df[df.columns[-1]]
        X = sm.add_constant(X)
        est = sm.OLS(y,X).fit()
        text_file.write(est.summary().as_text())

  text_file.close()
  text_file = open("figures/RegressionMultipleValues.txt", "a")
  # regression of all columns
  Z = df[df.columns.tolist()[:-1]]
  #t = df[df.columns[-1]]
  Z = sm.add_constant(Z)
  est2 = sm.OLS(y,Z).fit()
  text_file.write(est2.summary().as_text())
  text_file.close()


def kruskalWallis(df, alpha):

  print(" Kruskal Wallis H-test test:")
  h = list(df.columns.values)

  for column in h[:-1]:
     # get the H and pval
     H, pval = mstats.kruskalwallis(df[column].tolist(),df["quality"].tolist())

     print " H-statistic:", H
     print " P-Value:", pval
     #check pvalue
     if pval < alpha:
         print "Reject NULL hypothesis - Significant differences exist between ",column," and quality \n\n"
     if pval >= alpha:
         print "Accept NULL hypothesis - No significant difference between ", column," and quality \n\n"



def isNormalDistribution(df,alpha,shapiro=True):

   print "\nChecking if the columns follow a normal distribution by d'Agostino & Pearson or Shpapiro test...\n"
   #list of column except the "quality"
   h = list(df.columns.values)
   count = 0
   for i in  h:

       u,v = ss.shapiro(df[i])
       k,p = mstats.normaltest(df[i])

       if (shapiro):
           if v < alpha:
              print "   The null hypothesis can be rejected; Column: ", i,"\n"
              count += 1
           else: print "   The null hypothesis can not be rejected; Column: ",i,"\n"
           
       else:
           if p < alpha:
              print "   The null hypothesis can be rejected; Column: ", i,"\n"
              count += 1
           else: print "   The null hypothesis can not be rejected; Column: ",i,"\n"
       if count == len(h):
           print "\n\n   Any column follows a normal distribution\n"



def isHomogeneous(df,alpha,levene=True):


    print "\nChecking if all the columns are homogeneous by Levene or Fligner-Killeen test...\n"

    #colums to list
    h = list(df.columns.values)[:-1]
    #columns values to list
    col1 = df[h[0]].tolist()
    col2 = df[h[1]].tolist()
    col3 = df[h[2]].tolist()
    col4 = df[h[3]].tolist()
    col5 = df[h[4]].tolist()
    col6 = df[h[5]].tolist()
    col7 = df[h[6]].tolist()
    col8 = df[h[7]].tolist()
    col9 = df[h[8]].tolist()
    col10 = df[h[9]].tolist()
    col11 = df[h[10]].tolist()

    L,p_val = ss.levene(col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11)
    F, p = ss.fligner(col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11)

    if (levene):
        if p_val < alpha:

           print "\n   It is not an homegeneous dataset (Levene)\n"

        else: print "\n   It is an homogeneneous dataset (Levene)\n"  
    
    else:
        if p < alpha:

           print "\n   It is not an homegeneous dataset (Fligner-Killeen) \n"

        else: print "\n   It is an homogeneneous dataset (Fligner-Killen)\n"  


def regression(df, r):

  h = list(df.columns.values)

  val = []
  #we loop simetrical matrix
  for i in range (len(h)):


    for j in range(i+1,len(h)):

      #compute line parameters
      slope, intercept, rvalue, p, error = ss.linregress(df[h[i]],df[h[j]])
      #print rvalue**2, h[i],h[j]
      #if correlation is good enough ...
      if  (rvalue**2 > r or rvalue**2 < -r) :

         print "\n***  The line equation is: ", h[j] ," =",slope,"*", h[i]," + (",intercept,")      ***\n" 

         val.append((h[i],h[j]))

  count = 0
  #plotting the lines which are good enough
  for x in val:
         
      sns.lmplot(x=x[0],y=x[1],data=df.sample(n=30),scatter=True,fit_reg=True)
      title = "line"+str(count)+".png"

      plt.savefig('figures/'+title)
      count += 1
  plt.show(block=False)
  plt.clf()



def checkOutliers(df, maxQ, minQ,  applyFunction=True,removeOutliers=True):


 h = list(df.columns.values)[:-1]
 dp = pd.DataFrame(columns=h)
 totalOutliers = 0
 if (applyFunction):
     print "Counting the outliers..."
     print "Column---Number---Outliers Up---Outliers Down"
     for i in h:
		 outliers = 0
		 #compute iqr and maxQuantil and minQuantil
		 iqr = np.percentile(df[i].tolist(),maxQ)- np.percentile(df[i].tolist(),minQ)
		 maxQuantil = np.percentile(df[i].tolist(),maxQ) + float(iqr*1.5)
		 minQuantil = np.percentile(df[i].tolist(),minQ) - float(iqr*1.5)
		 supOutliers = df[i][df[i] > maxQuantil].count()
		 infOutliers = df[i][df[i] < minQuantil].count()
		 outliers += supOutliers + infOutliers
		 
		 
		 print i,"-->",outliers, ",", supOutliers, ",",infOutliers
		
		 #plotting the outliers
		 flierprops = dict(markerfacecolor='1.75', markersize=5,linestyle='none')
		 sns.boxplot(df[i],flierprops=flierprops)
		 plt.savefig('figures/'+str(i)+"_BoxPlot")
		 plt.show(block=False)
		 plt.clf()
		 
		 #converting the outliers to NaN values
		 df[i] = df[i][df[i] < maxQuantil]
		 df[i] = df[i][df[i] > minQuantil]
		 
     #remove the rows with NaN values
     dp = df.dropna(axis=0, how='any')

     totalOutliers = df[h[0]].count() - dp[h[0]].count()
     print "\n*** The total outliers in the dataset are :" +str(totalOutliers)+ "  ***\n "
     #if removeOutliers we remove the NaN rows.  If not we replace by the Nan value by the mean
     if (removeOutliers):
    
        return dp

     else: 
         dp = df.fillna(df.mean())
         return dp
 else:
     return df



def normalizedData(df):

  h = list(df.columns.values)
  dp = pd.DataFrame(columns=h)
  #we normalize data between 0 and 1
  for i in h:
     maxim = float(df[i].max())
     minim = float(df[i].min())
     dp[i] = df[i].map(lambda x: (x- minim)/ (maxim-minim))

  return dp

  
   
def drawNormal(df):

  h = list(df.columns.values)[:-1]
  dp = pd.DataFrame(columns=h)
  #we plot the histogrames and the normal curve.
  for i in h:
     v = sorted(df[i].tolist())
     fit = ss.norm.pdf(v,np.mean(v),np.std(v))
     plt.plot(v,fit)
     plt.hist(v,normed='True',label=i)
     s = i+".png"
     plt.savefig('figures/'+s)
     plt.show(block=False)
     plt.clf()
     
     # We also plot the Q-Q graphic of the mean
     s = i+"_Q-Q_plot.png" 
     mean = np.mean(df[i])
     std = np.std(df[i], ddof=1)
     dp[i] = df[i].map(lambda x:(x-mean)/std)
     ss.probplot(dp[i],plot=plt)
     plt.savefig('figures/'+s)
     plt.show(block=False)
     plt.clf()



if __name__=="__main__":
   
   #create the folder for storing the images
   createImagesFolder('figures')

   #reading the dataset and printing the basical statistics
   df = pd.read_csv("wine.csv",sep=';')
   df.describe().to_csv("wineStatistics.csv")

   # checking the outliers 
   dc = checkOutliers(df,75,25,False,True)


   #normalizing data
   #dn = normalizedData(dc)
   dn = dc
   #checking if normal distribution
   isNormalDistribution(dn,0.05)

   #checking homogeneousity
   isHomogeneous(dn,0.05)
   
   #applying Kruskall Wallis hypothesis
   kruskalWallis(dn,0.05)

   #computing the possible lines between fields given a correlation and plotting the lines
   regression(dn,0.675)
   
   #plotting the normal curves for each fiel
   drawNormal(dn)

   #applyin multiple regression for each pair of variable and for all the variables 
   multipleRegression(dn)

   # printing the final dataset statistics and the final dataset itself
   dn.to_csv("wineTreated.csv")
   temp = dn.describe()
   temp.to_csv('wineTreatedStatistics.csv')

   #uncomment if you want to apply pca 
   """
   applyPCA(dn,2)

   """
   














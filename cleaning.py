import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from scipy import stats as ss
import seaborn as sns
from scipy.stats import mstats
import os


def createImagesFolder(directory):

   if not os.path.exists(directory):
     os.makedirs(directory)


def kruskalWallis(df, alpha):

   print(" Kruskal Wallis H-test test:")
   # get the H and pval
   H, pval = mstats.kruskalwallis([df[col] for col in df.columns.tolist()[:-1]])

   print " H-statistic:", H
   print " P-Value:", pval
   #check pvalue
   if pval < alpha:
      print("Reject NULL hypothesis - Significant differences exist between groups.\n\n")
   if pval >= alpha:
      print("Accept NULL hypothesis - No significant difference between groups.\n\n")



def isNormalDistribution(df,alpha):

  print "\nChecking if it the columns follow a normal distribution by d'Agostino & Pearson test...\n"
  #list of column except the "quality"
  h = list(df.columns.values)[:-1]
  count = 0
  for i in  h:
    #u,v = ss.shapiro(df[i])

    k,p = mstats.normaltest(df[i])

    if p < alpha:
       print "   The null hypothesis can be rejected; Column: ", i,"\n"
       count += 1
    else: "   The null hypothesis can not be rejected; Column: ",i,"\n"
  if count == len(h):
       print "\n\n   Any column follows a normal distribution\n"


def isHomogeneous(df,alpha):


    print "\nChecking if all the columns are homogeneous by Levene test...\n"

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
    W,p_val = ss.levene(col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11)

    if p_val < alpha:

       print "\n   It is not an homegeneous dataset\n"

    else: print "\n   It is an homogeneneous dataset\n"  



def regression(df, r):

  h = list(df.columns.values)
  val = []
  #we loop simetrical matrix
  for i in range (len(h)):

    for j in range(i+1,len(h)):

      #compute line parameters
      slope, intercept, rvalue, p, error = ss.linregress(df[h[i]],df[h[j]])

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



def checkOutliers(df, maxQ, minQ, removeOutliers=True):


  h = list(df.columns.values)[:-1]
  dp = pd.DataFrame(columns=h)
  totalOutliers = 0
  for i in h:
     outliers = 0
     #compute iqr and maxQuantil and minQuantil
     iqr = np.percentile(df[i].tolist(),maxQ)- np.percentile(df[i].tolist(),minQ)
     maxQuantil = np.percentile(df[i].tolist(),maxQ) + float(iqr*1.5)
     minQuantil = np.percentile(df[i].tolist(),minQ) - float(iqr*1.5)
     outliers += df[i][df[i] > maxQuantil].count() + df[i][df[i] < minQuantil].count()
     # convert the outlier to NaN in dataset
     df[i] = df[i][df[i] < maxQuantil]
     df[i] = df[i][df[i] > minQuantil]

     #dp[i] =df[i].interpolate(method='polynomial')
     #dp[i] = df[i].fillna(df[i].mean())
  
  #remove the rows with NaN values
  dp = df.dropna(axis=0, how='any')

  totalOutliers = df[h[0]].count() - dp[h[0]].count()
  print "\n*** The total outliers in the dataset are :" +str(totalOutliers)+ "***\n "
  #if removeOutliers we remove the NaN rows.  If not we replace by the Nan value by the mean
  if (removeOutliers):
    return dp

  else: 
    dp = df.fillna(df.mean())
    return dp


def normalizedData(df):

  h = list(df.columns.values)
  dp = pd.DataFrame(columns=h)
  #we normalize data between 0 and 1
  for i in h:
    maxim = float(df[i].max())
    minim = float(df[i].min())
    dp[i] = df[i].map(lambda x: (x- minim)/ (maxim-minim))

  return dp

"""
def drawNormal2(df):

  h = list(df.columns.values)
  dp = pd.DataFrame(columns=h)
  
  for i in h[:-1]:
     mean = np.mean(df[i])
     std = np.std(df[i], ddof=1)
     dp[i] = df[i].map(lambda x:(x-mean)/std)
     dp[i].plot(kind='hist',title=i) 
     s = i+".png"
     plt.savefig(s)

     
     plt.show(block=False)
     plt.clf()

"""  
   
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
   dc = checkOutliers(df,75,25)

   #normalizing data
   dn = normalizedData(dc)

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

   #drawNormal2(dn)
   
   # printing the final dataset statistics and the final dataset itself
   dn.to_csv("wineTreated.csv")
   temp = dn.describe()
   temp.to_csv('out.csv')
   #np.savetxt('outtext.txt',df.describe(), fmt='%f')
   





















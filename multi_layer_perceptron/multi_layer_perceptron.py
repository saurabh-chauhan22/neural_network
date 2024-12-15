import logging

import pandas as pd 
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

logger = logging.getLogger(__name__)

class MultiLayerPerceptron:
    '''
    Multi-layer perceptron class for diabetes prediction dataset 
    '''
    def __init__(self, dataset_file) -> None:
        self.dataframe = pd.read_csv(dataset_file)
        
    def cleaned_data(self):
        '''
        Cleaning and prepossesing data 
        Return cleaned data
        ''' 
        pd.set_option("display.max_columns",500)
        print(self.dataframe.describe(include="all"))
        print("Number of rows with 0 values for each variable : ")
        for col in self.dataframe.columns:
            missing_rows = self.dataframe.loc[self.dataframe[col] == 0].shape[0]
            print(col +" : " + str(missing_rows))
        
        self.dataframe["Glucose"] = self.dataframe["Glucose"].replace(0, np.nan)
        self.dataframe["BloodPressure"] = self.dataframe["BloodPressure"].replace(0, np.nan)
        self.dataframe["SkinThickness"] = self.dataframe["SkinThickness"].replace(0, np.nan)
        self.dataframe["Insulin"] = self.dataframe["Insulin"].replace(0, np.nan)
        self.dataframe["BMI"] = self.dataframe["BMI"].replace(0, np.nan)
        

    def prepossessing_data(self):
        # Normalize the data via centering
        # Use the scale() function from scikit-learn
        print("Centering the data...")
        df_scaled = preprocessing.scale(self.dataframe)
        # Result must be converted back to a pandas DataFrame
        df_scaled = pd.DataFrame(df_scaled, columns=self.dataframe.columns)
        # Do not want the Outcome column to be scaled, so keep the original
        df_scaled['Outcome'] = self.dataframe['Outcome']
        df = df_scaled
        print(df.describe().loc[['mean', 'std','max'],].round(2).abs())
        
    def build_model(self):
        # Split dataset into an input matrix (all columns but Outcome) and Outcome vector
        X = self.dataframe.loc[:, self.dataframe.columns != 'Outcome']
        y = self.dataframe.loc[:, 'Outcome']
        # Split input matrix to create the training set (80%) and testing set (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Second split on training set to create the validation set (20% of training set)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=8))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=1, epochs=200)
        # Evaluate the accuracy with respect to the training set
        scores_1 = model.evaluate(X_train, y_train)
        print("Training Accuracy1: %.2f%%\n" % (scores_1[1]*100))
        # Evaluate the accuracy with respect to the testing set
        score_2 = model.evaluate(X_test, y_test)
        print("Testing Accuracy2: %.2f%%\n" % (score_2[1]*100))
        
        
        
        y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
        c_matrix = confusion_matrix(y_test, y_test_pred)
        ax = sns.heatmap(c_matrix, annot=True,
        xticklabels=['No Diabetes','Diabetes'],
        yticklabels=['No Diabetes','Diabetes'],
        cbar=False, cmap='Blues')
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        plt.show()
        
        y_test_pred_probs = model.predict(X_test)
        FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
        plt.plot(FPR, TPR)
        plt.plot([0,1],[0,1],'--', color='black') #diagonal line
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
          
        
    def explore_analysis(self)-> None:
        '''
        Explore the dataset and clean the irrreguralities and inconsistencies
        '''
        # self.dataframe.hist()
        self.dataframe.hist(bins=136)
        plt.show(block=True)
        
    def seaborn_plotting(self):
        '''
        Plot the seaborn plotting
        '''
        # Create a subplot of 3 x 3
        figure, axes = plt.subplots(3,3,figsize=(15,15))
        # Make sure there is enough padding to allow titles to be seen
        figure.tight_layout(pad=5.0)
        # Plot a density plot for each variable
        for idx, col in enumerate(self.dataframe.columns):
            ax = plt.subplot(3, 3, idx + 1)
            ax.yaxis.set_ticklabels([])
            sns.kdeplot(data=self.dataframe.loc[self.dataframe.Outcome == 0, col], ax=ax,
                    linestyle='-', color='black', label="No Diabetes")
            sns.kdeplot(data=self.dataframe.loc[self.dataframe.Outcome == 1, col], ax=ax,
                    linestyle='--', color='black', label="Diabetes")
            ax.set_title(col)
        plt.subplot(3,3,9).set_visible(False)
        plt.show()
          

if __name__ =='__main__':
    m_perceptron = MultiLayerPerceptron("diabetes.csv")
    m_perceptron.explore_analysis()
    m_perceptron.cleaned_data()
    m_perceptron.prepossessing_data()
    m_perceptron.seaborn_plotting()
    m_perceptron.build_model()
    

    
    
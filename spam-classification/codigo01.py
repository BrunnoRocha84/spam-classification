# Import of libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
 
print("Script started...")  # Debug message 

# loading dataset 

try: 
    df = pd.read_csv("C:/Users/tarcisio.rocha_onfly/Downloads/spam.csv", encoding='latin-1')  # I used enconding='latin-1' to avoid reading errors 
    print("Dataset loaded successfully!")  # Debug message 
except Exception as e: 
    print(f"Error loading dataset: {e}")  # Error message 

# checking the columns 
print("\nOriginal dataset columns:") 
print(df.columns) 
 # Renaming columns for ease of use 
df = df[['v1', 'v2']] 
df.columns = ['label', 'text'] 
print("\nDataset ater renaming columns:") 
print(df.head()) 
# Exploratory Data Analysis (EDA) 

try: 

    # Checking the distribution of classes (spam vs ham) 
    sns.countplot(x='label', data=df) 
    plt.title('Distribuição de Spam vs Ham') 
    plt.savefig('distribuicao_spam_ham.png')  # Save the graph as an image 
    plt.close()  # Close the figure to free up memory 
    print("Distribution chart saved as 'distribuicao_spam_ham.png'.") 

    # Checking the dataset size 
    print(f"\nNumber of emails: {len(df)}") 

    # Exemplo de e-mails spam e ham 
    print("\nSpam email example:") 
    print(df[df['label'] == 'spam']['text'].iloc[0]) 
    print("\nHam email example:") 
    print(df[df['label'] == 'ham']['text'].iloc[0]) 
    print("EDA successfully completed!")  # Debug message 

except Exception as e: 
    print(f"Error during a EDA: {e}")  # Error message 

# Data Preprocessing 

try: 

    # Converting labels to binary values (spam = 1, ham = 0) 
    df['label'] = df['label'].map({'spam': 1, 'ham': 0}) 
    # Splitting the dataset into training and testing 

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42) 

    # Text vectorization (Bag of Words) 
    vectorizer = CountVectorizer() 
    X_train_vec = vectorizer.fit_transform(X_train) 
    X_test_vec = vectorizer.transform(X_test) 

    print("Preprocessing completed sucessfully!")  # Debug message 

except Exception as e: 
    print(f"Error during the prepocessing: {e}")  # Debug message 

# Modeling 
try: 
    # Creating the Naive Bayes Model 
    nb_model = MultinomialNB() 

    # Training the model 
    nb_model.fit(X_train_vec, y_train) 

    # Makiing predictions 
    y_pred_nb = nb_model.predict(X_test_vec) 

    # Evaluating the model  
    accuracy_nb = accuracy_score(y_test, y_pred_nb) 
    print(f'\nNaive Bayes - Acurácia: {accuracy_nb:.2f}') 
    print(classification_report(y_test, y_pred_nb)) 
    print("Matriz de Confusão:") 
    print(confusion_matrix(y_test, y_pred_nb)) 

    # Saving the confusion matrix as an image 

    sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues') 
    plt.title('Matriz de Confusão - Naive Bayes') 
    plt.savefig('matriz_confusao_nb.png')  # Save the graph as an image 
    plt.close()  # Close the figure to free up memory 
    print("Confusion matrix saved as 'matriz_confusao_nb.png'.") 
    print("Modeling completed successfully!")  # Debug message 

except Exception as e: 
    print(f"Error during modeling: {e}")  # Error message 
print("Finished script.")  # Debug message 

 
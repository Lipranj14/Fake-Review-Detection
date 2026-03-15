import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def load_data(filepath="amazon_reviews.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Please run generate_dataset.py first.")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} reviews.")
    return df

def feature_engineering(df):
    print("Engineering features...")
    
    # 1. Review Length (number of words)
    df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
    
    # 2. Exclamation Count
    df['exclamation_count'] = df['review_text'].apply(lambda x: str(x).count('!'))
    
    # 3. Reviewer Review Count (how many reviews has this user left?)
    user_counts = df['user_id'].value_counts().to_dict()
    df['reviewer_review_count'] = df['user_id'].map(user_counts)
    
    # 4. Rating Deviation 
    # Calculate the average rating per product
    product_avg_rating = df.groupby('product_id')['rating'].mean().to_dict()
    # absolute deviation from the mean product rating
    df['rating_deviation'] = np.abs(df['rating'] - df['product_id'].map(product_avg_rating))
    
    return df

def apply_tfidf(df, max_features=500):
    print(f"Applying TF-IDF vectorization (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    # Fit and transform the review text
    tfidf_matrix = vectorizer.fit_transform(df['review_text'].fillna(""))
    
    # Create a DataFrame from the TF-IDF matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Save the vectorizer for the Streamlit app later
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    return tfidf_df

def plot_eda(df):
    """Generates distribution plots for the engineered features to visually prove they work."""
    print("Generating EDA visualizations...")
    os.makedirs("eda_plots", exist_ok=True)
    
    features_to_plot = ['review_length', 'exclamation_count', 'reviewer_review_count', 'rating_deviation']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        sns.boxplot(data=df, x='label', y=feature, ax=axes[i], palette="Set2")
        axes[i].set_title(f'Distribution of {feature}\n(0=Genuine, 1=Fake)')
        
    plt.tight_layout()
    plt.savefig("eda_plots/feature_distributions.png")
    print("EDA plots saved to eda_plots/feature_distributions.png")

def process_pipeline():
    # 1. Load Data
    df = load_data()
    
    # 2. Feature Engineering
    df_engineered = feature_engineering(df)
    
    # 3. EDA (Save plots)
    plot_eda(df_engineered)
    
    # 4. TF-IDF
    df_tfidf = apply_tfidf(df_engineered)
    
    # 5. Combine and Save Final Training Data
    # Drop columns we can't train on directly
    numeric_features = df_engineered[['review_length', 'exclamation_count', 'verified_purchase', 
                                     'reviewer_review_count', 'rating_deviation']]
                                     
    # Concat the numeric behavioral features with the TF-IDF NLP features
    X = pd.concat([numeric_features, df_tfidf], axis=1)
    y = df_engineered['label']
    
    print(f"Final feature matrix shape: {X.shape}")
    
    # Save to CSV for model_training.py
    X.to_csv("X_train_processed.csv", index=False)
    y.to_csv("y_train_processed.csv", index=False)
    print("Data processing complete. Saved X_train_processed.csv and y_train_processed.csv.")

if __name__ == "__main__":
    process_pipeline()

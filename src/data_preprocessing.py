# src/data_preprocessing.py
import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewPreprocessor:
    def __init__(self):
        # Indonesian slang dictionary for normalization
        self.slang_dict = {
            # Quality terms
            'bgus': 'bagus',
            'bgs': 'bagus',
            'mantap': 'bagus',
            'mantul': 'bagus',
            'oke': 'bagus',
            'ok': 'bagus',
            'jelek': 'buruk',
            'ancur': 'buruk',
            'rusak': 'buruk',
            'parah': 'buruk',
            
            # Shipping terms
            'cpt': 'cepat',
            'cpet': 'cepat',
            'kilat': 'cepat',
            'lama': 'lambat',
            'lelet': 'lambat',
            
            # Price terms
            'murmer': 'murah',
            'mumer': 'murah',
            'worth': 'sebanding',
            'worthit': 'sebanding',
            'mahil': 'mahal',
            
            # Service terms
            'rekom': 'rekomendasi',
            'recommended': 'rekomendasi',
            'respon': 'responsif',
            'fast': 'cepat',
            'slow': 'lambat',
            
            # General terms
            'bgt': 'banget',
            'bgt': 'banget',
            'bener': 'benar',
            'gak': 'tidak',
            'ga': 'tidak',
            'tp': 'tapi',
            'jd': 'jadi',
            'krn': 'karena',
            'udh': 'sudah',
            'udah': 'sudah',
            'blm': 'belum',
            'belom': 'belum'
        }
        
        # Aspect keywords for multi-label classification
        self.aspect_keywords = {
            'kualitas_produk': {
                'positive': ['bagus', 'berkualitas', 'original', 'asli', 'premium', 'excellent', 'kualitas', 'mantap'],
                'negative': ['jelek', 'buruk', 'rusak', 'kw', 'palsu', 'fake', 'cacat', 'ancur', 'parah']
            },
            'harga': {
                'positive': ['murah', 'worth', 'sebanding', 'value', 'affordable', 'terjangkau'],
                'negative': ['mahal', 'overprice', 'kemahalan', 'expensive', 'pricey']
            },
            'pengiriman': {
                'positive': ['cepat', 'kilat', 'express', 'fast', 'tepat', 'ontime'],
                'negative': ['lama', 'lambat', 'telat', 'slow', 'delay', 'lelet']
            },
            'pelayanan': {
                'positive': ['ramah', 'baik', 'responsif', 'helpful', 'fast', 'respon', 'sopan'],
                'negative': ['buruk', 'jelek', 'tidak', 'slow', 'lambat', 'cuek', 'galak']
            },
            'performa': {
                'positive': ['battery', 'speed', 'cepat', 'lancar', 'smooth', 'camera', 'bagus', 'performance'],
                'negative': ['lemot', 'lag', 'hang', 'error', 'lambat', 'boros', 'panas', 'overheat']
            },
            'packaging': {
                'positive': ['rapi', 'aman', 'bubble', 'wrap', 'packaging', 'bungkus', 'kemasan'],
                'negative': ['rusak', 'jelek', 'buruk', 'hancur', 'penyok', 'lecek']
            }
        }
    
    def clean_text(self, text):
        """Clean and normalize Indonesian text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (but keep price-related ones)
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters but keep indonesian characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize slang
        words = text.split()
        normalized_words = []
        for word in words:
            if word in self.slang_dict:
                normalized_words.append(self.slang_dict[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def create_aspect_labels(self, df):
        """Create multi-label aspect classification labels"""
        logger.info("Creating aspect labels...")
        
        for aspect, keywords in self.aspect_keywords.items():
            # Initialize with zeros
            df[f'has_{aspect}'] = 0
            
            # Check for aspect mentions
            for _, row in df.iterrows():
                text = row['clean_text'].lower()
                
                # Check if any keyword is present
                has_positive = any(keyword in text for keyword in keywords['positive'])
                has_negative = any(keyword in text for keyword in keywords['negative'])
                
                if has_positive or has_negative:
                    df.at[row.name, f'has_{aspect}'] = 1
        
        return df
    
    def create_sentiment_labels(self, df):
        """Create sentiment labels for each aspect"""
        logger.info("Creating sentiment labels...")
        
        for aspect, keywords in self.aspect_keywords.items():
            # Initialize with neutral (0)
            df[f'sentiment_{aspect}'] = 0
            
            for _, row in df.iterrows():
                text = row['clean_text'].lower()
                
                # Count positive and negative keywords
                positive_count = sum(1 for keyword in keywords['positive'] if keyword in text)
                negative_count = sum(1 for keyword in keywords['negative'] if keyword in text)
                
                # Determine sentiment
                if positive_count > negative_count:
                    df.at[row.name, f'sentiment_{aspect}'] = 1  # Positive
                elif negative_count > positive_count:
                    df.at[row.name, f'sentiment_{aspect}'] = -1  # Negative
                # else remains 0 (neutral)
        
        return df
    
    def preprocess_data(self, input_file='data/raw/tokopedia_reviews.csv', 
                    output_file='data/processed/processed_reviews.csv'):
        """Main preprocessing pipeline"""
        logger.info(f"Loading data from {input_file}")
        
        # Load data
        df = pd.read_csv(input_file)

        # All data
        print(f"Number of the data: {df.shape}")

        # --- HAPUS DATA YANG KOSONG DI 'ulasan' DAN 'nama_barang' ---
        df.dropna(subset=['ulasan', 'nama_barang'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Rename to match expected column name
        df.rename(columns={'ulasan': 'review_text'}, inplace=True)
        
        # Clean text
        logger.info("Cleaning text...")
        df['clean_text'] = df['review_text'].apply(self.clean_text)
        
        # Remove empty reviews (after cleaning)
        df = df[df['clean_text'].str.strip().str.len() > 0]
        
        # Create aspect labels
        df = self.create_aspect_labels(df)
        
        # Create sentiment labels
        df = self.create_sentiment_labels(df)
        
        # Add text length feature
        df['text_length'] = df['clean_text'].str.len()
        
        # Add word count feature
        df['word_count'] = df['clean_text'].str.split().str.len()
        
        # Save processed data
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        return df
    
    def analyze_data(self, df):
        """Analyze the processed data"""
        print("\n=== Data Analysis ===")
        print(f"Total reviews: {len(df)}")
        print(f"Average text length: {df['text_length'].mean():.2f}")
        print(f"Average word count: {df['word_count'].mean():.2f}")
        
        print("\n=== Aspect Distribution ===")
        aspect_columns = [col for col in df.columns if col.startswith('has_')]
        for col in aspect_columns:
            aspect_name = col.replace('has_', '')
            count = df[col].sum()
            percentage = (count / len(df)) * 100
            print(f"{aspect_name}: {count} reviews ({percentage:.1f}%)")
        
        print("\n=== Rating Distribution ===")
        print(df['rating'].value_counts().sort_index())
        
        print("\n=== Sample Processed Data ===")
        sample_cols = ['clean_text', 'rating'] + aspect_columns[:3]
        print(df[sample_cols].head())
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        logger.info("Splitting data...")
        
        # Features and labels
        X = df['clean_text']
        y = df[[col for col in df.columns if col.startswith('has_')]]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y.iloc[:, 0]
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        # Save splits
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv('data/processed/train_data.csv', index=False)
        val_df.to_csv('data/processed/val_data.csv', index=False)
        test_df.to_csv('data/processed/test_data.csv', index=False)
        
        print(f"Data split completed:")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df

def main():
    """Main function for data preprocessing"""
    preprocessor = ReviewPreprocessor()
    
    # Preprocess data
    df = preprocessor.preprocess_data()
    
    # Analyze data
    df = preprocessor.analyze_data(df)
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df)
    
    return df

if __name__ == "__main__":
    df = main()
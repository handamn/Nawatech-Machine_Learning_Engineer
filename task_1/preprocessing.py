
import pandas as pd
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class IndonesianTextPreprocessor:
    def __init__(self, use_sastrawi=True):
        self.use_sastrawi = use_sastrawi
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

        # Indonesian stopwords (manual backup if Sastrawi not available)
        self.manual_stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'oleh',
            'adalah', 'akan', 'ada', 'atau', 'juga', 'ini', 'itu', 'tidak', 'sudah', 'bisa',
            'ya', 'aja', 'sih', 'deh', 'dong', 'kan', 'lah', 'kok', 'saja', 'jadi', 'kalau',
            'kalo', 'udah', 'udh', 'gak', 'ga', 'nggak', 'enggak', 'banget', 'bgt', 'sich'
        }
        
        # Custom stopwords for twit
        self.custom_stopwords = {'rt', 'user', 'mention', 'link', 'url'}
    
    # Clean spesific alias tokens
    def clean_anonymized_tokens(self, text):
        if pd.isna(text):
            return ""
        
        # Remove user_mention and url
        text = re.sub(r'<USER_MENTION>', '', text)
        text = re.sub(r'<URL>', '', text)
        
        # Replace to provider and produk
        text = text.replace('<PROVIDER_NAME>', 'provider')
        text = text.replace('<PRODUCT_NAME>', 'produk')
        
        return text
    
    # Clean Text
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        # lowercase preprocess
        text = text.lower()
        
        # hashtags preprocess to keep the content
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove RT (retweet indicator)
        text = re.sub(r'^rt\s+', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and keep Indonesian char
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        return text.strip()
    
    # Remove stopwords
    def remove_stopwords(self, text):
        if pd.isna(text) or text.strip() == "":
            return ""
        
        if self.use_sastrawi:
            # Use Sastrawi stopword remover
            text = self.stopword_remover.remove(text)
        else:
            # Manual stopword removal
            words = text.split()
            words = [word for word in words if word not in self.manual_stopwords]
            text = ' '.join(words)
        
        # Remove custom stopwords
        words = text.split()
        words = [word for word in words if word not in self.custom_stopwords]
        
        return ' '.join(words)
    
    # Tokenize text to word
    def tokenize_text(self, text):
        if pd.isna(text) or text.strip() == "":
            return []
        
        # Split by whitespace and filter out empty strings
        tokens = [token.strip() for token in text.split() if token.strip()]
        
        # Filter out very short tokens (< 2 characters)
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    # Stemming
    def stem_text(self, text):
        if pd.isna(text) or text.strip() == "":
            return ""
        
        if self.use_sastrawi:
            return self.stemmer.stem(text)
        else:
            # Simple suffix removal for common Indonesian suffixes
            words = text.split()
            stemmed_words = []
            for word in words:
                # Remove common suffixes
                if word.endswith('kan'):
                    word = word[:-3]
                elif word.endswith('an'):
                    word = word[:-2]
                elif word.endswith('nya'):
                    word = word[:-3]
                stemmed_words.append(word)
            return ' '.join(stemmed_words)
    
    # Do preprocess
    def preprocess(self, text, return_tokens=False):
        # Step 1: Handle anonymized tokens
        text = self.clean_anonymized_tokens(text)
        
        # Step 2: Basic cleaning
        text = self.clean_text(text)
        
        # Step 3: Remove stopwords
        text = self.remove_stopwords(text)
        
        # Step 4: Stemming
        text = self.stem_text(text)
        
        # Step 5: Final cleanup
        text = ' '.join(text.split())  # Remove extra spaces
        
        # Step 6: Tokenization
        if return_tokens:
            return self.tokenize_text(text)
        
        return text

class ExploreAndPrepPipeline:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.csv_path = f"{self.dataset_dir}/dataset_tweet_sentiment_cellular_service_provider.csv"
        self.df = None
        self.preprocessor = IndonesianTextPreprocessor()
    
    def load_data(self):
        print("=== Loading Dataset ===")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    # dataset information
    def explore_data(self):
        if self.df is None:
            print("Please load data first!")
            return
        
        print("\n=== Dataset Overview ===")
        print(self.df.head())
        
        print("\n=== Data Info ===")
        print(self.df.info())
        
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
        
        print("\n=== Sentiment Distribution ===")
        sentiment_counts = self.df['Sentiment'].value_counts()
        print(sentiment_counts)
        
        # Calculate percentages
        sentiment_pct = self.df['Sentiment'].value_counts(normalize=True) * 100
        print("\nSentiment Percentages:")
        for sentiment, pct in sentiment_pct.items():
            print(f"{sentiment}: {pct:.1f}%")
    
    # Create visualizations for data information
    def visualize_data(self, save_plots=True):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sentiment Distribution
        sentiment_counts = self.df['Sentiment'].value_counts()
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, 
                       color=['red', 'green'])
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].set_xlabel('Sentiment')
        axes[0, 0].set_ylabel('Count')
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts.values):
            axes[0, 0].text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # 2. Text Length Distribution
        text_lengths = self.df['Text Tweet'].str.len()
        axes[0, 1].hist(text_lengths, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].set_xlabel('Character Count')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Text Length by Sentiment
        for sentiment in self.df['Sentiment'].unique():
            subset = self.df[self.df['Sentiment'] == sentiment]
            axes[1, 0].hist(subset['Text Tweet'].str.len(), alpha=0.7, 
                           label=sentiment, bins=15)
        axes[1, 0].set_title('Text Length by Sentiment')
        axes[1, 0].set_xlabel('Character Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Word Count Distribution
        word_counts = self.df['Text Tweet'].str.split().str.len()
        axes[1, 1].hist(word_counts, bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_title('Word Count Distribution')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{self.dataset_dir}/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def validate_preprocessing(self):
        print("\n=== Preprocessing Validation ===")
        
        # Check for empty texts after preprocessing
        empty_count = self.df['processed_text'].str.strip().eq('').sum()
        print(f"Empty texts after preprocessing: {empty_count}")
        
        # Check for empty token lists
        empty_tokens = self.df['tokens'].apply(lambda x: len(x) == 0).sum()
        print(f"Empty token lists: {empty_tokens}")
        
        # Check average length before/after
        avg_length_before = self.df['Text Tweet'].str.len().mean()
        avg_length_after = self.df['processed_text'].str.len().mean()
        print(f"Average character length before: {avg_length_before:.2f}")
        print(f"Average character length after: {avg_length_after:.2f}")
        
        # Check word count before/after
        avg_words_before = self.df['Text Tweet'].str.split().str.len().mean()
        avg_words_after = self.df['processed_text'].str.split().str.len().mean()
        avg_tokens = self.df['tokens'].apply(len).mean()
        print(f"Average word count before: {avg_words_before:.2f}")
        print(f"Average word count after: {avg_words_after:.2f}")
        print(f"Average token count: {avg_tokens:.2f}")
        
        # Show reduction percentage
        reduction_pct = ((avg_length_before - avg_length_after) / avg_length_before) * 100
        print(f"Text length reduction: {reduction_pct:.1f}%")
        
        # Check for very short texts (might be problematic)
        very_short = self.df['processed_text'].str.len() < 5
        print(f"Very short texts (< 5 chars): {very_short.sum()}")
        
        # Token statistics
        print(f"\n=== Token Statistics ===")
        all_tokens = [token for token_list in self.df['tokens'] for token in token_list]
        unique_tokens = len(set(all_tokens))
        print(f"Total tokens: {len(all_tokens)}")
        print(f"Unique tokens: {unique_tokens}")
        print(f"Vocabulary richness: {unique_tokens/len(all_tokens):.4f}")
        
        # Show token length distribution
        token_lengths = [len(token) for token in all_tokens]
        print(f"Average token length: {np.mean(token_lengths):.2f}")
        print(f"Min token length: {min(token_lengths) if token_lengths else 0}")
        print(f"Max token length: {max(token_lengths) if token_lengths else 0}")
    
    # apply preprocessing
    def apply_preprocessing(self):
        print("\n=== Applying Preprocessing ===")
        
        # Show sample before preprocessing
        print("Sample data BEFORE preprocessing:")
        for i in range(min(3, len(self.df))):
            print(f"ID {self.df.iloc[i]['Id']}: {self.df.iloc[i]['Text Tweet']}")
        
        # Apply preprocessing
        print("\nProcessing texts...")
        self.df['processed_text'] = self.df['Text Tweet'].apply(
            lambda x: self.preprocessor.preprocess(x)
        )
        
        # Apply tokenization and store as list of tokens
        print("Tokenizing texts...")
        self.df['tokens'] = self.df['Text Tweet'].apply(
            lambda x: self.preprocessor.preprocess(x, return_tokens=True)
        )
        
        # Show sample after preprocessing
        print("\nSample data AFTER preprocessing:")
        for i in range(min(3, len(self.df))):
            print(f"ID {self.df.iloc[i]['Id']}:")
            print(f"  Processed: {self.df.iloc[i]['processed_text']}")
            print(f"  Tokens: {self.df.iloc[i]['tokens']}")
        
        # Validation
        self.validate_preprocessing()
    
    #Generate word clouds for positive and negative sentiments
    def generate_wordcloud(self, save_plots=True):
    
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Positive sentiment word cloud
        positive_texts = ' '.join(self.df[self.df['Sentiment'] == 'positive']['processed_text'])
        if positive_texts.strip():
            wordcloud_pos = WordCloud(width=800, height=400, 
                                     background_color='white', colormap='viridis').generate(positive_texts)
            axes[0].imshow(wordcloud_pos, interpolation='bilinear')
            axes[0].set_title('Positive Sentiment Word Cloud', fontsize=16)
            axes[0].axis('off')
        
        # Negative sentiment word cloud
        negative_texts = ' '.join(self.df[self.df['Sentiment'] == 'negative']['processed_text'])
        if negative_texts.strip():
            wordcloud_neg = WordCloud(width=800, height=400, 
                                     background_color='white', colormap='Reds').generate(negative_texts)
            axes[1].imshow(wordcloud_neg, interpolation='bilinear')
            axes[1].set_title('Negative Sentiment Word Cloud', fontsize=16)
            axes[1].axis('off')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{self.dataset_dir}/wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    #Get top words for each sentiment
    def get_top_words(self, n=10):
        print(f"\n=== Top {n} Words by Sentiment ===")
        
        for sentiment in self.df['Sentiment'].unique():
            subset = self.df[self.df['Sentiment'] == sentiment]
            all_words = ' '.join(subset['processed_text']).split()
            word_freq = Counter(all_words)
            
            print(f"\n{sentiment.upper()} Sentiment:")
            for word, freq in word_freq.most_common(n):
                if word.strip():  # Skip empty words
                    print(f"  {word}: {freq}")
    
    def save_processed_data(self, name_file='processed_sentiment_data.csv'):
        output_path = f'{self.dataset_dir}/{name_file}'
        
        # Save with original columns plus processed text and tokens
        output_df = self.df[['Id', 'Sentiment', 'Text Tweet', 'processed_text']].copy()
        
        # Convert tokens list to string for CSV storage
        output_df['tokens_str'] = self.df['tokens'].apply(lambda x: ' '.join(x))
        
        output_df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        
        # Also save tokens as separate file for advanced usage
        tokens_df = self.df[['Id', 'Sentiment', 'tokens']].copy()
        tokens_output_path = output_path.replace('.csv', '_tokens.csv')
        tokens_df.to_csv(tokens_output_path, index=False)
        print(f"Tokens data saved to: {tokens_output_path}")
    
    def run_complete_analysis(self):
        print("Starting Complete Sentiment Analysis Pipeline")
        print("=" * 50)
        
        # Step 1: Load data
        if not self.load_data():
            return
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Visualize data
        self.visualize_data()
        
        # Step 4: Apply preprocessing
        self.apply_preprocessing()
        
        # Step 5: Generate word clouds
        self.generate_wordcloud()
        
        # Step 6: Get top words
        self.get_top_words()
        
        # Step 7: Save processed data
        self.save_processed_data()
        
        print("Complete analysis pipeline finished!")

if __name__ == "__main__":
    dataset_dir = "dataset"

    # Initialize analyzer with your CSV file path
    analyzer = ExploreAndPrepPipeline(dataset_dir)
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
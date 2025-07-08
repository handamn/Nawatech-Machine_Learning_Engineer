import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import ast

class SentimentInsightVisualizer:
    def __init__(self, data_path='dataset/processed_sentiment_data_tokens.csv'):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        if self.df['tokens'].dtype == object:
            self.df['tokens'] = self.df['tokens'].apply(lambda x: ast.literal_eval(x))
    
    # Sentiment Distribution
    def plot_sentiment_distribution(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x='Sentiment', palette='Set2')
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("insight/sentiment_distribution.png", dpi=300)
        plt.close()
        print("Sentiment distribution saved.")

    #Wordcloud
    def generate_wordclouds(self):
        sentiments = self.df['Sentiment'].unique()
        for sent in sentiments:
            text = ' '.join([' '.join(tokens) for tokens in self.df[self.df['Sentiment'] == sent]['tokens']])
            wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis' if sent.lower() == 'positive' else 'Reds').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Word Cloud - {sent.capitalize()}")
            plt.tight_layout()
            plt.savefig(f"insight/wordcloud_{sent.lower()}.png", dpi=300)
            plt.close()
            print(f"WordCloud for '{sent}' saved.")
    
    #Most Frequent Words
    def most_frequent_words(self, top_n=20):
        sentiments = self.df['Sentiment'].unique()
        for sent in sentiments:
            tokens = [token for tokens in self.df[self.df['Sentiment'] == sent]['tokens'] for token in tokens]
            common = Counter(tokens).most_common(top_n)
            words, freqs = zip(*common)
            plt.figure(figsize=(10, 4))
            sns.barplot(x=list(freqs), y=list(words), palette='viridis')
            plt.title(f"Top {top_n} Words - {sent.capitalize()} Sentiment")
            plt.xlabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"insight/top_words_{sent.lower()}.png", dpi=300)
            plt.close()
            print(f"Top words for '{sent}' saved.")

    #Complain Category Analysis
    def complaint_category_analysis(self, keywords=None):
        #manual setup keyword
        if keywords is None:
            keywords = ['sinyal', 'jaringan', 'kuota', 'paket', 'internet', 'harga', 'cs', 'pelayanan']

        results = {k: 0 for k in keywords}
        for tokens in self.df['tokens']:
            for k in keywords:
                if k in tokens:
                    results[k] += 1

        keys = list(results.keys())
        vals = list(results.values())
        plt.figure(figsize=(10, 4))
        sns.barplot(x=vals, y=keys, palette='magma')
        plt.title("Complaint Category Frequency")
        plt.xlabel("Mentions")
        plt.tight_layout()
        plt.savefig("insight/complaint_categories.png", dpi=300)
        plt.close()
        print("Complaint category analysis saved.")

if __name__ == '__main__':
    visualizer = SentimentInsightVisualizer()
    visualizer.load_data()
    visualizer.plot_sentiment_distribution()
    visualizer.generate_wordclouds()
    visualizer.most_frequent_words()
    visualizer.complaint_category_analysis()

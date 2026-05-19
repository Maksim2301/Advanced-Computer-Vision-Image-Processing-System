import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import time
import urllib3
import random
from nltk.stem import WordNetLemmatizer
import spacy
import numpy as np

# Вимикаємо попередження SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def setup_nlp():
    print("Завантаження ресурсів NLTK...")
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

lemmatizer = WordNetLemmatizer()
setup_nlp()

try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    print("Помилка: Модель spaCy не знайдена!")
    print("Будь ласка, виконайте в терміналі: python -m spacy download en_core_web_sm")
    exit()

# --- 1. WEB-СКРАПІНГ ---

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.google.com/'
}


def fetch_html(url):
    try:
        time.sleep(random.uniform(0.5, 1.5))
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Помилка завантаження {url}: {e}")
        return None


def get_article_text(url, source):
    html = fetch_html(url)
    if not html: return ""
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['style', 'script', 'nav', 'footer', 'figcaption', 'aside', 'header', 'figure']):
        tag.decompose()

    container = None
    if source == 'The Guardian':
        container = soup.find('div', {'data-gu-name': 'body'}) or soup.find('article') or soup.find('main')
    elif source == 'NPR':
        container = soup.find('div', id='storytext') or soup.find('div', class_='storytext') or soup.find('article')
    elif source == 'PBS':
        container = soup.find('article') or soup.find(class_='body-text') or soup.find(class_='post-content')

    search_area = container if container else soup
    paragraphs = [p.get_text(strip=True) for p in search_area.find_all('p') if len(p.get_text(strip=True)) > 50]

    full_text = " ".join(paragraphs)
    return full_text[:3500]


def scrape_guardian(limit=50):
    print(f"Скрапінг The Guardian (Ціль: {limit} статей)...")
    news_list = []
    seen_links = set()
    page = 1

    while len(news_list) < limit and page <= 10:
        url = f'https://www.theguardian.com/world?page={page}'
        html = fetch_html(url)
        if not html: break

        soup = BeautifulSoup(html, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']

            if '/world/' in link and '/video/' not in link and '/audio/' not in link and '/gallery/' not in link and len(link.split('/')) > 5:
                if link.startswith('/'): link = 'https://www.theguardian.com' + link

                title = a_tag.get('aria-label') or a_tag.get_text(strip=True)

                if title and len(title) > 15 and link not in seen_links:
                    seen_links.add(link)
                    article_text = get_article_text(link, 'The Guardian')

                    if len(article_text) > 300:
                        news_list.append({
                            'source': 'The Guardian',
                            'title': title,
                            'description': article_text,
                            'link': link
                        })

            if len(news_list) >= limit: break
        page += 1

    print(f"The Guardian: Зібрано {len(news_list)}")
    return news_list


def scrape_npr(limit=50):
    print(f"Скрапінг NPR (Ціль: {limit} статей)...")
    sections = ['news/', 'world/', 'politics/', 'national/', 'business/']
    news_list = []
    seen_links = set()

    for section in sections:
        if len(news_list) >= limit: break
        url = f'https://www.npr.org/sections/{section}'
        html = fetch_html(url)
        if not html: continue

        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('article', class_='item')

        for article in articles:
            title_tag = article.find('h2', class_='title')
            if not title_tag or not title_tag.find('a'): continue
            title = title_tag.get_text(strip=True)
            link = title_tag.find('a').get('href', '')

            if len(title) > 20 and link and link not in seen_links:
                seen_links.add(link)
                article_text = get_article_text(link, 'NPR')
                if len(article_text) > 300:
                    news_list.append({'source': 'NPR', 'title': title, 'description': article_text, 'link': link})

            if len(news_list) >= limit: break

    print(f"NPR: Зібрано {len(news_list)}")
    return news_list


def scrape_pbs(limit=50):
    print(f"Скрапінг PBS NewsHour (Ціль: {limit} статей)...")
    news_list = []
    seen_links = set()
    page = 1

    while len(news_list) < limit and page <= 10:
        url = f'https://www.pbs.org/newshour/latest/page/{page}'
        html = fetch_html(url)
        if not html: break

        soup = BeautifulSoup(html, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']

            if '/newshour/' in link and len(link) > 40 and not link.endswith('/'):
                title = a_tag.get_text(strip=True)

                if len(title) > 20 and link not in seen_links:
                    seen_links.add(link)
                    if link.startswith('/'): link = 'https://www.pbs.org' + link

                    article_text = get_article_text(link, 'PBS')
                    if len(article_text) > 300:
                        news_list.append({'source': 'PBS', 'title': title, 'description': article_text, 'link': link})

            if len(news_list) >= limit: break
        page += 1

    print(f"PBS: Зібрано {len(news_list)}")
    return news_list


# --- 2. ОЧИЩЕННЯ ДАНИХ ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace('-', '')
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    custom_stops = {
        'said', 'news', 'new', 'time', 'people', 'one', 'would', 'also', 'told', 'could', 'year', 'two', 'first',
        'npr', 'pbs', 'guardian', 'caption', 'photo', 'image', 'images', 'getty', 'ap', 'reuters', 'press',
        'associated', 'imageshide', 'hide', 'orb', 'episode', 'medication',
        'nprhide', 'associate', 'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    }

    # Обробка тексту через spaCy
    doc = nlp(text)
    words = [
        token.lemma_ for token in doc
        if not token.is_stop and token.lemma_ not in custom_stops and len(token.lemma_) > 2
    ]

    return " ".join(words)


# --- 3. ПОРІВНЯЛЬНИЙ АНАЛІЗ ТА NLP ---
def analyze_and_visualize(df):
    print("\n--- Початок аналізу даних ---")
    df['full_text'] = df['description']
    df['cleaned_text'] = df['full_text'].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=max(2, int(0.05 * len(df))),
        max_df=0.85,
        ngram_range=(1, 1)
    )
    X = vectorizer.fit_transform(df['cleaned_text'])

    plt.figure(figsize=(15, 10))
    sources = df['source'].unique()
    top_words_per_source = {}

    for i, source in enumerate(sources, 1):
        source_idx = df.index[df['source'] == source].tolist()
        source_X = X[source_idx]
        sum_tfidf = source_X.sum(axis=0)

        words_freq = [(word, sum_tfidf[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_df = pd.DataFrame(words_freq, columns=['word', 'score'])
        words_df = words_df.sort_values(by='score', ascending=False).head(15)
        top_words_per_source[source] = words_df['word'].tolist()

        plt.subplot(2, 2, i)
        sns.barplot(x='score', y='word', data=words_df, hue='word', palette='viridis', legend=False)
        plt.title(f'Топ-15 слів: {source}')
        plt.xlabel('Вага TF-IDF')
        plt.ylabel('')

    plt.tight_layout()
    plt.savefig('top_words.png')
    print("Графік 'top_words.png' збережено.")
    return top_words_per_source


# --- 4. АНАЛІЗ ТОНАЛЬНОСТІ ---

def sentiment_analysis(df):
    sid = SentimentIntensityAnalyzer()

    def get_average_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.0

        sentence_scores = [sid.polarity_scores(sentence)['compound'] for sentence in sentences]
        return np.mean(sentence_scores)

    df['compound'] = df['full_text'].apply(get_average_sentiment)

    def categorize_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_type'] = df['compound'].apply(categorize_sentiment)

    stats = df.groupby('source').agg(
        avg_sentiment=('compound', 'mean'),
        total=('compound', 'count')
    )

    sent_counts = df.groupby(['source', 'sentiment_type']).size().unstack(fill_value=0)
    for col in ['Positive', 'Negative', 'Neutral']:
        if col not in sent_counts: sent_counts[col] = 0

    stats['pos_percent'] = (sent_counts['Positive'] / stats['total']) * 100
    stats['neg_percent'] = (sent_counts['Negative'] / stats['total']) * 100
    stats['neu_percent'] = (sent_counts['Neutral'] / stats['total']) * 100

    return stats

def generate_conclusion(top_words, sentiment_stats):
    print("\n1. Домінуючі теми (Ключові слова з відсіюванням аномалій):")
    for source, words in top_words.items():
        print(f"   - {source}: {', '.join(words[:7])}")

    print("\n2. Аналіз тональності:")
    for source, row in sentiment_stats.iterrows():
        print(f"   - {source}: Середня тональність {row['avg_sentiment']:.3f}. "
              f"(Поз: {row['pos_percent']:.1f}%, Нег: {row['neg_percent']:.1f}%, Нейтр: {row['neu_percent']:.1f}%)")

    print("\n3. Порівняння:")
    most_neg = sentiment_stats['neg_percent'].idxmax()
    most_pos = sentiment_stats['pos_percent'].idxmax()

    print(f"   - Найвища концентрація негативу: {most_neg}.")
    print(f"   - Найвища концентрація позитиву/оптимізму: {most_pos}.")
    most_balanced = sentiment_stats['avg_sentiment'].abs().idxmin()

    print(f"   - Найбільш нейтральне/збалансоване джерело (за середнім показником): {most_balanced}.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    setup_nlp()
    LIMIT = 50
    data_guardian = scrape_guardian(LIMIT)
    data_npr = scrape_npr(LIMIT)
    data_pbs = scrape_pbs(LIMIT)
    all_news = data_guardian + data_npr + data_pbs
    df_all = pd.DataFrame(all_news)
    counts = df_all['source'].value_counts()
    print("\nРозподіл зібраних статей за джерелами:")
    print(counts)
    min_count = counts.min()
    df_all = df_all.groupby('source').sample(min_count, random_state=42).reset_index(drop=True)
    if counts.min() < 20:
        print("\nНедостатньо даних для коректного порівняльного аналізу.")
        print("Потрібно мінімум 20 статей на джерело.")
        exit()

    print(f"\nВибірка збалансована до {min_count} статей на джерело.")

    if df_all.empty or len(counts) < 3:
        print("Не вдалося зібрати достатньо даних з усіх джерел. Перевірте з'єднання або HTML-структуру.")
    else:
        df_all.to_csv('all_news_merged.csv', index=False)
        print("Дані успішно збережено у CSV.")

        top_words = analyze_and_visualize(df_all)
        sentiment_stats = sentiment_analysis(df_all)
        generate_conclusion(top_words, sentiment_stats)
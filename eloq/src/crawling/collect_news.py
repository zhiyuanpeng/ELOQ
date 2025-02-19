import os
import json
from newscatcher import Newscatcher, urls
import feedparser
from newspaper import Article
from datetime import datetime
import time
from tqdm import tqdm

# Function to create necessary directories
def create_dirs(base_dir, topic):
    dir_path = os.path.join(base_dir, topic)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# Function to count existing JSON files in the topic directory
def count_existing_files(topic_dir):
    return len([name for name in os.listdir(topic_dir) if name.endswith('.json')])

# Function to collect articles for a topic
def collect_articles_for_topic(topic, max_articles=1000, dataset_name='News1k2024', cutt_off_date='2024-09-31', max_tokens=150):
    cutt_off_year, cutt_off_month, cutt_off_day = map(int, cutt_off_date.split('-'))
    base_dir = os.path.join(os.getcwd(), 'data', 'raw', dataset_name)

    # Create the base directory for the topic
    topic_dir = create_dirs(base_dir, topic)
    
    # Count existing JSON files
    existing_files = count_existing_files(topic_dir)
    
    # Skip if max articles are already collected
    if existing_files >= max_articles:
        print(f"\nTopic '{topic}' already has {existing_files} articles, skipping.")
        return
    
    # Determine starting index for new file names
    start_index = existing_files + 1
    
    print(f"\nCollecting articles for topic: {topic} (starting from article {start_index})")

    # Get the list of websites for the topic
    websites = urls(topic=topic, language='en')
    total_websites = len(websites)
    
    if not websites:
        print(f"No websites found for topic: {topic}")
        return

    # Progress bar for websites
    with tqdm(total=total_websites, desc="Websites", unit="site", position=0) as pbar_websites:
        collected_articles = existing_files
        for website in websites:
            try:
                nc = Newscatcher(website=website, topic=topic)
                results = nc.get_news()
                if not results or 'articles' not in results:
                    pbar_websites.update(1)
                    continue
                
                feed = results['articles']
                if not feed:
                    pbar_websites.update(1)
                    continue
                
                # Progress bar for feeds
                with tqdm(total=len(feed), desc=f"Feeds from {website}", unit="feed", position=1, leave=False) as pbar_feed:
                    for entry in feed:
                        if collected_articles >= max_articles:
                            return

                        # Parse publication date
                        published = entry.get('published_parsed') or entry.get('updated_parsed')
                        if not published:
                            pbar_feed.update(1)
                            continue
                        
                        published_date = datetime.fromtimestamp(time.mktime(published))
                        # Skip articles published before cutt-off date
                        if published_date <= datetime(cutt_off_year, cutt_off_month, cutt_off_day):
                            pbar_feed.update(1)
                            continue
                        
                        article_url = entry.get('link')
                        if not article_url:
                            pbar_feed.update(1)
                            continue
                        
                        # Use Newspaper3k to download and parse the article
                        article = Article(article_url)
                        try:
                            article.download()
                            article.parse()
                        except Exception as e:
                            pbar_feed.update(1)
                            continue
                        
                        # Check if content has at least 150 tokens
                        content_tokens = len(article.text.split())
                        if content_tokens < max_tokens:
                            pbar_feed.update(1)
                            continue
                        
                        # Prepare the data
                        article_data = {
                            'website': website,
                            'url': article_url,
                            'date': published_date.isoformat(),
                            'title': article.title or entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'content': article.text,
                            'author': article.authors,
                            'token_count': content_tokens
                        }

                        # Save the article as a JSON file
                        article_filename = os.path.join(topic_dir, f"{topic}_{collected_articles + 1}.json")
                        with open(article_filename, 'w', encoding='utf-8') as f:
                            json.dump(article_data, f, ensure_ascii=False, indent=4)
                        
                        collected_articles += 1
                        pbar_feed.update(1)  # Update the feed progress bar

                        # Update the main progress bar to show collected articles count
                        pbar_websites.set_postfix({'Collected Articles': collected_articles})
                        
                        # Check if max articles are collected
                        if collected_articles >= max_articles:
                            pbar_feed.close()
                            pbar_websites.close()
                            return
                    
                    pbar_feed.close()  # Close feed progress bar when done
            except Exception as e:
                # Uncomment the next line to debug exceptions
                # print(f"Error processing website {website}: {e}")
                pass  # Continue to next website regardless of errors
            finally:
                pbar_websites.update(1)  # Update the websites progress bar
        
        pbar_websites.close()  # Close websites progress bar when done

# Main execution
if __name__ == "__main__":
    # List of topics
    topics = [
        'news', 'business', 'science', 'food',
        'politics', 'travel', 'entertainment',
        'music', 'sport', 'tech'
    ]
    # other possible topis are: 'finance', 'economics', 'world'
    for topic in topics:
        collect_articles_for_topic(topic, max_articles=1000, dataset_name='News1k2024', cutt_off_date='2024-09-31', max_tokens=150)
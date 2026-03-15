import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Create a synthetic 5,000 row dataset for Amazon reviews to train our model on.
# This prevents downloading a massive 500MB Kaggle dataset for the demo.
def generate_amazon_reviews(num_rows=5000):
    print(f"Generating synthetic dataset with {num_rows} rows...")
    
    # Base dictionary
    data = {"review_id": [], "product_id": [], "user_id": [], "timestamp": [], 
            "review_text": [], "rating": [], "verified_purchase": [], "label": []}
    
    products = [f"PROD_{i}" for i in range(1, 101)] # 100 fake products
    users = [f"USER_{i}" for i in range(1, 4001)] # 4000 users
    
    # Generic review templates
    good_reviews = [
        "Absolutely love this product! Works exactly as described.",
        "Very high quality. I highly recommend buying this.",
        "Good value for the price. Would buy again.",
        "Arrived quickly and in perfect condition. Five stars.",
        "Solid product, does what it says on the box.",
        "My family loves it. Will be ordering more soon."
    ]
    bad_reviews = [
        "Terrible quality, broke after one use.",
        "Do not buy this. Complete waste of money.",
        "Arrived damaged and customer service was unhelpful.",
        "Not what the picture showed at all. Disappointed.",
        "Very overpriced for what you get.",
        "Stopped working within a week. Avoid."
    ]
    
    # Fake review templates (Overly exclamatory, weird length)
    fake_good_reviews = [
        "BEST PRODUCT EVER IN THE WORLD!!!!!!! I LOVE IT SO MUCH BUY THIS NOW BOUGHT 10!!!!!",
        "Amazing!!!!!! Incredible quality!!!! Absolutely perfect in every single way imaginable!!! A+++++++++",
        "Wow wow wow! So good! Great seller!",
        "Yes.",  # Suspiciously short
        "Perfect item. Love it."
    ]
    
    fake_bad_reviews = [
        "WORST THING I HAVE EVER SEEN IN MY LIFE DO NOT BUY SCAM SCAM SCAM SCAM!!!!!!",
        "Awful. Terrible. Hate it. Ruined my life.",
        "No."
    ]
    
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_rows):
        # 80% genuine, 20% fake
        is_fake = random.random() < 0.2
        
        user = random.choice(users)
        product = random.choice(products)
        
        # Fake reviews are much more likely to be unverified
        if is_fake:
            verified_purchase = 1 if random.random() < 0.1 else 0 # 10% verified
            # Extreme ratings
            rating = 5 if random.random() < 0.8 else 1 
            text = random.choice(fake_good_reviews) if rating == 5 else random.choice(fake_bad_reviews)
            label = 1 # 1 = Fake
        else:
            verified_purchase = 1 if random.random() < 0.85 else 0 # 85% verified
            # Normal distribution of ratings
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            text = random.choice(good_reviews) if rating >= 3 else random.choice(bad_reviews)
            label = 0 # 0 = Genuine
        
        # Add random noise/length to text to make TFIDF work
        extra_words = " " + " ".join(random.choices(["and", "the", "it", "was", "very", "so"], k=random.randint(0, 5)))
        text = text + extra_words
        
        timestamp = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        
        data["review_id"].append(f"REV_{i}")
        data["product_id"].append(product)
        data["user_id"].append(user)
        data["timestamp"].append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        data["review_text"].append(text)
        data["rating"].append(rating)
        data["verified_purchase"].append(verified_purchase)
        data["label"].append(label)
        
    df = pd.DataFrame(data)
    df.to_csv("amazon_reviews.csv", index=False)
    print("Dataset saved to amazon_reviews.csv")
    
if __name__ == "__main__":
    generate_amazon_reviews(5000)

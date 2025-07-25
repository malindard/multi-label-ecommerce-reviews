import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

urls = [
    'https://www.tokopedia.com/ismile-indonesia/review',
    'https://www.tokopedia.com/samsung-official-store/review',
    'https://www.tokopedia.com/oppo/review',
    'https://www.tokopedia.com/xiaomi/review',
    'https://www.tokopedia.com/vivo/review'
]

all_data = []

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

for url in urls:
    print(f"Processing: {url}")
    driver.get(url)

    nama_toko = url.split("/")[-2]

    for _ in range(400):
        try:
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))
            )
        except:
            print("The 'Next page' button was not found.")
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.find_all('article', class_='css-1pr2lii')

        for container in containers:
            # Reviews
            review_elem = container.find('span', attrs={'data-testid': 'lblItemUlasan'})
            review = review_elem.text.strip() if review_elem else None

            # Rating
            try:
                rating_elem = container.find('div', {'data-testid': 'icnStarRating'})
                rating_text = rating_elem.get('aria-label') if rating_elem else None
                rating = ''.join(filter(str.isdigit, rating_text)) if rating_text else None
            except:
                rating = None

            # Product name
            produk_elem = container.find('p', class_='css-akhxpb-unf-heading e1qvo2ff8')
            produk = produk_elem.text.strip() if produk_elem else None

            all_data.append((nama_toko, produk, review, rating))

        # Click the "Next page" button
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
            if next_button.is_enabled():
                next_button.click()
                time.sleep(5)
            else:
                break
        except:
            break

    print(f"Scraping completed for: {url}\n")

driver.quit()
print("All scraping done.")

# Create DataFrame
df = pd.DataFrame(all_data, columns=["toko", "nama_barang", "ulasan", "rating"])

# Save to CSV
df.to_csv("data/raw/tokopedia_reviews.csv", index=False, encoding='utf-8-sig')
print("Scraping completed for all URLs and data saved in 'data/raw/tokopedia_reviews.csv'")
import requests as r
from bs4 import BeautifulSoup
import pandas as pd

url="http://books.toscrape.com/"

response=r.get(url)
response.raise_for_status()
soup=BeautifulSoup(response.text,'html.parser')
books=soup.find_all('article',class_='product_pod')
book_data=[]


for i in books:
    title=i.h3.a['title']
    price=i.find('p',class_='price_color').text
    book_data.append({'Title':title,'Price':price})

df = pd.DataFrame(book_data)
df.to_csv('books.csv', index=False)

print("Scraping completed. Data saved to books.csv.")    
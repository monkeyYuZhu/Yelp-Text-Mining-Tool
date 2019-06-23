#HTML scraper

from bs4 import BeautifulSoup
import re
import time
import requests

#Get reviews restaurant names
def getreviews(urls):
    pageNum = 100  #100 pages
    restaurants = []
    file = open(urls)
    for line in file:
        restaurants.append(line.strip())
    file.close()

    fw = open('italian_html.txt', 'w')
    for restaurant in restaurants:
        page=restaurant
        print(page)
        for p in range(0, pageNum):
            print ('page', p)
            html = None
            if p == 0:
                pageLink = page  # url for page 1
            else:
                pageLink = page.replace("osq=italian", "start=") + str(p*2) + '0'
            for i in range(5):
                try:
                    response = requests.get(pageLink, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                    html = response.content
                    break
                except Exception as e:
                    print ('failed attempt', i)
                    time.sleep(2)  # wait 2 secs
            if not html: continue

            soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml')
            reviews = soup.findAll('div', {'class': re.compile('review-content')})
            names = soup.findAll('div', {'class': re.compile('biz-page-header-left')})
            if reviews=='':
                break
            for part in names:
                nameChunk = part.find('h1', {'class': re.compile('biz-page-title')})
                nameChunk = str(nameChunk)

                for part in reviews:
                    ratingChunk = part.find('p', {'lang': re.compile('en')})
                    ratingChunk = str(ratingChunk)

                    fw.write(nameChunk + '\t' + ratingChunk + '\n')  # write to file
                time.sleep(2)
    fw.close()

urls = 'italian_restaurants.txt'
getreviews(urls)


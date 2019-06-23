"""
@author: Christina Eng, Yu Zhu
"""

# ------------------------------------Function to get the data from the Yelp.com------------------------------------ #
# Get urls (An Italian example)
# Collect 50 italian restaurant URLs
from bs4 import BeautifulSoup
import requests, time
import re

def run(url):
    
    pageNum = 1

    for p in range(0, pageNum):  # for each page since the pagelink changes
        print ('page', p)
        html = None
        if p == 0:
            pageLink = url + str(p)  + '&cflt=italian,restaurants&ed_attrs=PlatformDelivery&l=p:DC:Washington::%5B,Downtown%5D' # url for page 1
        else:
            pageLink = url + str(p) + str(0) + '&cflt=italian,restaurants&ed_attrs=PlatformDelivery&l=p:DC:Washington::%5B,Downtown%5D' #url for other pages

        for i in range(5):
            try:
                response = requests.get(pageLink, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                html = response.content
                break
            except Exception as e:
                print ('failed attempt', i)
                time.sleep(1)
        if not html: continue
        
        htmllist = []
        soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml')
        for result in soup.findAll("span", {"class": re.compile('indexed-biz-name')}):
            rest = 'NA'

            business_url = "http://www.yelp.com" + result.find("a")["href"]
            if business_url: rest = business_url
            htmllist.append(rest)
            
            time.sleep(2)

    return htmllist
    
# Get the url
url = 'https://www.yelp.com/search?find_desc=italian&start=' #beginning of url that is the same for every restaurant
restaurant_url = run(url)


#------------------------------------------------------------------------------
########################################################
# The following functions can be applied to retrieve   #
# one specific restaurant's reviews. User's input is   #
# assumed to be the first page of the review.          #
########################################################

# The function to get the content of the review
def getContent(n):
    review = 'NA'
    contentChunk = n.find('p', {'lang': 'en'})
    if contentChunk:
        review = contentChunk.text
    return review

def getName(n):
    name = 'NA'
    nameChunk = n.find('h1', re.compile('biz-page-title embossed-text-white'))
    if nameChunk:
        name= nameChunk.text
    return name

# Scrape website with 'beautifulsoup'
# This url is the one that introduces every detail of one specific restaurant
def review_scraper(restaurant_url):
    
    pageNum = 1 # Number of page to collect
    
    review_list = []
    
    for p in range(1, pageNum + 1):
        print('page', p)
        html = None
        
        if p == 1:
            pageLink = restaurant_url
        else:
            pageLink = restaurant_url + '?start=' + str(20*(p-1)) # Make the url link
        
        for i in range(5): # Try 5 times
            try:
                # Use the brower to access the url
                response = requests.get(pageLink, headers = { 'User-Agent': 
                    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
                html = response.content # Get the html
                break # When got the file, break the loop
            except Exception as e: # browser.open() threw an exception, the attempt to get the response failed
                print('failed attempt', i)
                time.sleep(3) # Wait 2 secs
                
        if not html:
            continue # Couldn't get the page, ignore
            
        soup = BeautifulSoup(html.decode('ascii', 'ignore'), 'lxml') # Parse the html 'html.parser'
        reviews = soup.findAll('div', {'class': 'review-content'})
        getname = soup.findAll('div', {'class': 'biz-page-header clearfix'})
        
        for n in getname:
            name = (getName(n)).strip()
        
        for n in reviews:
            review = getContent(n)
            review_list.append(review)
            time.sleep(2)
        
    return name, review_list


# ------------------------------------Put the previous data into a dictionary------------------------------------ #

# This will be the end product: a dictionary whose key is the restaurant name and value is all of its reviews
review_dict = {}

# Get reviews
for i in range(len(restaurant_url)):
    name, review_list = review_scraper(restaurant_url[i])
    review_dict[name] = review_list

# After you get the value of review_dict
# italian_dict = review_dict 
# and do the same thing to assign value to chinese_dict and mexican_dict


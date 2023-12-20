import scrapy
from scrapy.selector import Selector 

import json

class NewsItem(scrapy.Item):
    article_urls = scrapy.Field()
    article_titles = scrapy.Field()
    article_subtitles = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()

class MothershipSpider(scrapy.Spider):
    name = 'mothershipbot'

    start_urls = ['https://mothership.sg/']
    load_more_start = "https://mothership.sg/json/posts-" 
    load_more_end = ".json"
    
    end_page = 26
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:77.0) Gecko/20100101 Firefox/77.0'}
    
    def parse(self, response):
        articles = response.css("[class^='ind-article']").getall()
        
        valid_str = "https://mothership.sg/"
        valid_url = []
        
        for article in articles:
            selArticle = Selector(text = article)
            url = selArticle.css("*::attr(href)").get()
        
            #Some cleaning rules
            if not url in valid_url:   
                full_url = response.urljoin(url)
                
                if valid_str in full_url:
                    valid_url.append(url)
                    yield scrapy.Request(full_url, callback = self.parse_article, headers = self.headers)
        
        for page_num in range(1, self.end_page):
            req_str = self.load_more_start + str(page_num) + self.load_more_end
            yield scrapy.Request(req_str, callback = self.parse_more, headers = self.headers)

    def parse_article(self, response):
        selHeader = Selector(text = response.css("div[class='header']").get())
        selImg = Selector(text = response.css("figure[class='featured-image']").get())
        
        scraped_info = NewsItem()
        scraped_info['article_urls'] = response.url

        scraped_info['article_titles'] = selHeader.css("h1::text").get()
        scraped_info['article_subtitles'] = selHeader.css("[class='subtitle']::text").get()
        
        scraped_info['image_urls'] = [response.urljoin(selImg.css("img::attr(src)").get())]
         
        yield scraped_info
        
    def parse_more(self, response):
        data = json.loads(response.body)
        
        valid_str = "https://mothership.sg/"
        valid_url = []
        
        for item in data:
            #print(item)
            #wait = input("Press Enter to continue.")
            
            url = item["url"]
            #Some cleaning rules
            if valid_str in url:
                if not url in valid_url:
                    valid_url.append(url)
                    
                    full_url = response.urljoin(url)
                    yield scrapy.Request(full_url, callback = self.parse_article, headers = self.headers)
import scrapy
from scrapy.selector import Selector 

import json

class CnaItem(scrapy.Item):
    article_urls = scrapy.Field()
    article_titles = scrapy.Field()
    image_captions = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()

class CnabotMoreSpider(scrapy.Spider):
    name = 'cnabotMore'
    #allowed_domains = ['www.channelnewsasia.com/']
    
    #Singapore
    #start_urls = ['https://www.channelnewsasia.com/singapore']
    #view_more = "https://www.channelnewsasia.com/api/v1/infinitelisting/94f7cd75-c28b-4c0a-8d21-09c6ba3dd3fc?_format=json&viewMode=infinite_scroll_listing&page="
    
    #Asia
    #start_urls = ['https://www.channelnewsasia.com/asia']
    #view_more = "https://www.channelnewsasia.com/api/v1/infinitelisting/1da7e932-70b3-4a2e-891f-88f7dd72c9d6?_format=json&viewMode=infinite_scroll_listing&page="
    
    #World
    #start_urls = ['https://www.channelnewsasia.com/world']
    #view_more = "https://www.channelnewsasia.com/api/v1/infinitelisting/9f7462b9-d170-42c1-a26c-5f89720ff5c9?_format=json&viewMode=infinite_scroll_listing&page="
    
    #Business
    start_urls = ['https://www.channelnewsasia.com/business']
    view_more = "https://www.channelnewsasia.com/api/v1/infinitelisting/5207efc4-baf1-47a8-a6d3-47a940cc115c?_format=json&viewMode=infinite_scroll_listing&page="
    
    #COVID-19
    #start_urls = ['https://www.channelnewsasia.com/coronavirus-covid-19']
    #view_more = "https://www.channelnewsasia.com/api/v1/infinitelisting/ddeeb87a-fa8d-4ae7-a685-f4ff13b0ddfb?_format=json&viewMode=infinite_scroll_listing&page="
    
    end_page = 20
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0'}
    
    def parse(self, response):
        urls = response.css("*::attr(href)").getall()
        
        valid_cat = ["/world/","/asia/","/singapore/","/business/","/commentary/","/women/","/sport/","/entertainment/"]
        
        valid_url = []
        
        for url in urls:
            #Some cleaning rules
            if any(url.startswith(s) for s in valid_cat):
                if not url in valid_url:
                    valid_url.append(url)
                    
                    full_url = response.urljoin(url)
                    yield scrapy.Request(full_url, callback = self.parse_article, headers = self.headers)
        
        for page_num in range(1, self.end_page):
            req_str = self.view_more + str(page_num)
            yield scrapy.Request(req_str, callback = self.parse_more, headers = self.headers)

    def parse_article(self, response):
        selArt = Selector(text = response.css("article").get())
        selCap = Selector(text = selArt.css("figcaption").get())
        
        scraped_info = CnaItem()
        scraped_info['article_urls'] = response.url
        scraped_info['image_urls'] = [response.urljoin(selArt.css("img::attr(src)").get())]
        scraped_info['article_titles'] = selArt.css("img::attr(title)").get()
        
        caption = selCap.css("p::text").get()
        if not caption:
            caption = selCap.css("*::text").get()
            
        if caption:
            caption = " ".join(caption.split())
        else:
            caption = ""   
            
        scraped_info['image_captions'] = caption
         
        yield scraped_info
        
    def parse_more(self, response):
        data = json.loads(response.body)
        
        for article in data["result"]:
            url = article["absolute_url"]
            yield scrapy.Request(url, callback = self.parse_article, headers = self.headers)
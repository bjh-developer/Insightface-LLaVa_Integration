import scrapy
from scrapy.selector import Selector 

import json

class NewsItem(scrapy.Item):
    article_urls = scrapy.Field()
    article_titles = scrapy.Field()
    image_captions = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()

class CoconutsSpider(scrapy.Spider):
    name = 'coconutsbot'

    start_urls = ['https://coconuts.co/singapore/news/']
    load_more_start = "https://coconuts.co/wp-admin/admin-ajax.php?action=load_more_ajax&paged=" 
    load_more_end = "&city=&category=700008&tags=&neighborhoods=&search_string=&post_type=&order_by="
    
    end_page = 10
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:77.0) Gecko/20100101 Firefox/77.0'}
    
    def parse(self, response):
        articles = response.css("[class='coco-article-title']").getall()
        valid_str = "/singapore/news/"
        
        valid_url = []
        
        for article in articles:
            selArticle = Selector(text = article)
            url = selArticle.css("*::attr(href)").get()
        
            #Some cleaning rules
            if valid_str in url:
                if not url in valid_url:
                    valid_url.append(url)
                    
                    full_url = response.urljoin(url)
                    yield scrapy.Request(full_url, callback = self.parse_article, headers = self.headers)
        
        for page_num in range(2, self.end_page):
            req_str = self.load_more_start + str(page_num) + self.load_more_end
            yield scrapy.Request(req_str, callback = self.parse_more, headers = self.headers)

    def parse_article(self, response):
        selBody = Selector(text = response.css("body").get())
        selImg = Selector(text = selBody.css("figure[class='post-image']").get())
       
        scraped_info = NewsItem()
        scraped_info['article_urls'] = response.url
        scraped_info['image_urls'] = [response.urljoin(selImg.css("img::attr(data-lazy-src)").get())]
        scraped_info['article_titles'] = selBody.css("[class='post-title']::text").get()
        
        caption = selImg.css("figcaption::text").get()
        if caption:
            caption = " ".join(caption.split())
        else:
            caption = ""
            
        scraped_info['image_captions'] = caption
         
        yield scraped_info
        
    def parse_more(self, response):
        data = json.loads(response.body)
        selMore = Selector(text = data["results"])
        
        valid_str = "/singapore/news/"
        valid_url = []
        
        articles = selMore.css("[class='coco-article-title']").getall()
        for article in articles:
            selArticle = Selector(text = article)
            url = selArticle.css("*::attr(href)").get()
        
            #Some cleaning rules
            if valid_str in url:
                if not url in valid_url:
                    valid_url.append(url)
                    
                    full_url = response.urljoin(url)
                    yield scrapy.Request(full_url, callback = self.parse_article, headers = self.headers)
import scrapy
from scrapy.selector import Selector 

import json

class NewsItem(scrapy.Item):
    article_urls = scrapy.Field()
    article_titles = scrapy.Field()
    image_captions = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()

class AsiaOneSpider(scrapy.Spider):
    name = 'asiaonebot'
    #allowed_domains = ['www.channelnewsasia.com/']
    
    start_urls = ["https://www.asiaone.com/"]
    req_start = "https://www.asiaone.com/jsonapi/node/article?page[offset]="
    req_end = "&page[limit]=10&fields[node--article]=title,created,field_source,field_rotator_headline,field_category,field_image,path,drupal_internal__nid&fields[file--file]=image_style_uri&fields[taxonomy_term--source]=name&fields[taxonomy_term--category]=name&include=field_source,field_image,field_category&sort=-field_publication_date&filter[category][condition][path]=field_category.drupal_internal__tid&filter[category][condition][value]=2&filter[status][condition][path]=status&filter[status][condition][value]=1"
    
    end_page = 1000
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:77.0) Gecko/20100101 Firefox/77.0'}
    
    def parse(self, response):
        curr_page = 0
        while curr_page < self.end_page:
            req_str = self.req_start + str(curr_page) + self.req_end
            yield scrapy.Request(req_str, callback = self.parse_more, headers = self.headers)
            
            curr_page = curr_page + 10

    def parse_article(self, response):
        selImg = Selector(text = response.css("div[class='image']").get())
    
        scraped_info = NewsItem()
        scraped_info['article_urls'] = response.url

        scraped_info['article_titles'] = response.css("[class='title']::text").get()
        
        scraped_info['image_urls'] = [response.urljoin(selImg.css("img::attr(src)").get())]
        
        caption = selImg.css("div[class='image-caption']::text").get()   
        if caption:
            caption = " ".join(caption.split())
        else:
            caption = ""   
        scraped_info['image_captions'] = caption
         
        yield scraped_info
        
    def parse_more(self, response):
        data = json.loads(response.body)
        
        for article in data["data"]:
            url = article["attributes"]["path"]["alias"]
            full_url = response.urljoin(url)
            yield scrapy.Request(full_url, callback = self.parse_article, headers = self.headers)
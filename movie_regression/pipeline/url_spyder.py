
# coding: utf-8

# In[ ]:


import scrapy


class UrlSpider(scrapy.Spider):
    name = 'get_urls'

    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "CONCURENT_REQUESTS_PER_DOMAIN": 3,
        "HTTPCACHE_ENABLED": True
    }
    
    start_urls = ['https://www.movieguide.org/category/reviews']
    
    def parse(self, response):

        urls = []
        for href in response.xpath(
            '//*[@class="cb-mask"]/a/@href'
        ).extract():
            if (
                ('interviews' not in href) and
                ('watch-reviews' not in href) and
                ('news-articles' not in href)
            ):
                yield {
                    'url' : href
                }

        next_url = response.xpath(
                '//*[@id="cb-blog-infinite-scroll"]/a/@href'
                ).extract()[0]
        
        yield scrapy.Request(
            url=next_url,
            callback=self.parse
        )       


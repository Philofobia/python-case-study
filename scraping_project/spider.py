import scrapy


class GamesSpider(scrapy.Spider):
    name = "games"
    start_urls = ['https://www.instant-gaming.com/it/pc/steam/in-tendenza/']

    def parse(self, response):
        for games in response.css('div.listing-items'):
            try:
                yield {
                    'title': games.css('div.information div.text div.name span.title::text').get(),
                    'dlc': games.css('div.information div.text div.name span.dlc::text').get(),
                    'price': games.css('div.information div.text div.price::text').get(),
                    'discount': games.css('a.cover div.discount::text').get(),
                    'link':  games.css('a.cover::attr(href)').get(),
                }
            except Exception:
                yield {
                    'title': games.css('div.information div.text div.name span.title::text').get(),
                    'dlc': 'Base Game',
                    'price': '-',
                    'discount': '0%',
                    'link': games.css('a.cover::attr(href)').get(),
                }

        next_page = response.css(
            'ul.pagination li a.arrow.right::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)


# generate a json with all the data
# scrapy runspider spider.py -o games.json
